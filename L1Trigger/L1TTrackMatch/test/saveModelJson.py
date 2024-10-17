import conifer
import joblib
xgb = joblib.load("/nfs/data41/rmccarth/conifer/vertexing/dispVertTaggerEmulationFixedPoint.pkl")
model = conifer.converters.convert_from_xgboost(xgb)
model.save('dispVertTaggerEmulationFixedPoint.json')
