import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hcalRecHitsDQMClient = DQMEDHarvester("HcalRecHitsDQMClient", 
#     outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
