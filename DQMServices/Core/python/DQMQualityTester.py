from FWCore.ParameterSet.Config import EDAnalyzer
def DQMQualityTester(*args, **kwargs):
  return EDAnalyzer("QualityTester", *args, **kwargs)
