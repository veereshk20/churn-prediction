import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

#Import the model
model = load_model("model.h5")

#Load all the pickle files

with open("gender_encoder.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    label_encoder_geo = pickle.load(file)

with open("scalar.pkl", "rb") as file:
    scalar = pickle.load(file)


## Streamlit app

st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score= st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card ?', [0, 1])
is_active_member = st.selectbox('Is active member ?', [0, 1])

#Prepare the input for the model
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

input_data["Gender"] = label_encoder_gender.transform(input_data["Gender"])
geo_encoded = label_encoder_geo.transform(input_data[["Geography"]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=label_encoder_geo.get_feature_names_out(["Geography"]))
input_data = pd.concat([input_data.drop(["Geography"], axis=1), geo_encoded_df], axis=1)
input_data = scalar.transform(input_data)


#Predict the Churn
prediction = model.predict(input_data)
prediction_pro = prediction[0][0]

st.write(f"Probabilty:{prediction_pro:.2f}")

if prediction_pro > 0.5:
    st.write("Customer is likely to churn.")
else:
    st.write("Customer is likely to not churn")

