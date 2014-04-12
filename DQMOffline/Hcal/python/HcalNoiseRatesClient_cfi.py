import FWCore.ParameterSet.Config as cms

hcalNoiseRatesClient = cms.EDAnalyzer("HcalNoiseRatesClient", 
#     outputFile = cms.untracked.string('NoiseRatesHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
