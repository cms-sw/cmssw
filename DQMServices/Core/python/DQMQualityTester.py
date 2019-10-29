import FWCore.ParameterSet.Config as cms
def DQMQualityTester(*args, **kwargs):
  return cms.EDAnalyzer("QualityTester", *args, **kwargs)
