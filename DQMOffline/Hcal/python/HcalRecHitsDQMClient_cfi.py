import FWCore.ParameterSet.Config as cms

hcalRecHitsDQMClient = cms.EDProducer("HcalRecHitsDQMClient", 
#     outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
