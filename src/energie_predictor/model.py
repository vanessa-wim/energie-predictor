import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


# Experiment benennen
mlflow.set_experiment("energie-predictor")

# Trainingsdaten generieren
X, y = make_regression(n_samples=100, n_features=1, noise=10)

# MLflow Experiment starten
with mlflow.start_run():

    # Modell trainieren
    model = LinearRegression()
    model.fit(X, y)

    # Vorhersagen machen
    predictions = model.predict(X)

    # Fehler berechnen
    mse = mean_squared_error(y, predictions)

    # Alles in MLflow speichern
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"MSE: {mse}")