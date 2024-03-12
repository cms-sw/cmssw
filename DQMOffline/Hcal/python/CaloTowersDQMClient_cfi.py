import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

calotowersDQMClient = DQMEDHarvester("CaloTowersDQMClient",
#     outputFile = cms.untracked.string('CaloTowersHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
# foo bar baz
# D7qsaLs4mDG8r
# EZ9ydYj2M1SDv
