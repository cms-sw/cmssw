import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hcalNoiseRatesClient = DQMEDHarvester("HcalNoiseRatesClient", 
#     outputFile = cms.untracked.string('NoiseRatesHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
