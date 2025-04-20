import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('oh.pkl','rb') as file:
    oh = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Customer Chrun Prediction')

geography = st.selectbox("Geography",oh.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age",18,100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
number_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])

# Submit Button
submit_button = st.button("Submit")

# Create the DataFrame when the button is clicked
if submit_button:
    input_data = pd.DataFrame({
        'CreditScore' : [credit_score],
        'Geography': [geography],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [number_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]

    })

    geo_encoded = oh.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded,columns=oh.get_feature_names_out(['Geography']))

    input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
    input_data.drop(columns=['Geography'],inplace=True)

    scaled_input = scaler.transform(input_data)

    pred = model.predict(scaled_input)
    pred_prob = pred[0][0]

    st.write(f'Chrun Probability: {pred_prob:.2f}')

    if pred_prob > 0.5 :
        st.write("The customer is likely to chrun.")
    else:
        st.write("The customer is not likely to chrun.")