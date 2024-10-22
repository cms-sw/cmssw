import FWCore.ParameterSet.Config as cms
def DQMQualityTester(*args, **kwargs):
  return cms.EDProducer("QualityTester", 
    inputGeneration = cms.untracked.string("DQMGenerationHarvesting"),
    outputGeneration = cms.untracked.string("DQMGenerationQTest"),
    *args, **kwargs
  )
