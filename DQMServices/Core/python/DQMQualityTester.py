from FWCore.ParameterSet.Config import EDProducer
def DQMQualityTester(*args, **kwargs):
  return EDProducer("QualityTester", *args, **kwargs)
