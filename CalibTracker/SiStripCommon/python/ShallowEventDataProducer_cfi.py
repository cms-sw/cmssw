import FWCore.ParameterSet.Config as cms

shallowEventRun = cms.EDProducer(
   "ShallowEventDataProducer",
   trigRecord = cms.InputTag('gtDigis'),
   lumiScalers = cms.InputTag("scalersRawToDigi"),
   metadata = cms.InputTag('onlineMetaDataDigis')
   )
