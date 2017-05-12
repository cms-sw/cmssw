import FWCore.ParameterSet.Config as cms

calotowersDQMClient = cms.EDProducer("CaloTowersDQMClient",
#     outputFile = cms.untracked.string('CaloTowersHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
