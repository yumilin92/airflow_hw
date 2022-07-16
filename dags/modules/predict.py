from datetime import datetime
import pandas as pd
import dill
import json
import os


path = os.environ.get('PROJECT_PATH', '/Users/yuliv/airflow/dags')


def get_raw_test_data():
    test_dir = f'{path}/dags/data/test/'
    test_data = []
    for filename in os.listdir(test_dir):
        with open(test_dir + filename, 'rb') as f:
            form = json.load(f)
            test_data.append(form)
    return test_data


def predict():
    with open(f'{path}/dags/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    test_data = get_raw_test_data()

    df = pd.DataFrame.from_dict(test_data)
    df['predict'] = model.predict(df)
    df.to_csv(f'{path}/dags/data/predictions/preds{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()