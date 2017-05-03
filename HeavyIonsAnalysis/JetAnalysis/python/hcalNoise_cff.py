import FWCore.ParameterSet.Config as cms


hcalNoise = cms.EDAnalyzer("HiHcalAnalyzer",
                           NoiseSummaryTag = cms.untracked.InputTag("hcalnoise"),
                           NoiseRBXTag = cms.untracked.InputTag("hcalnoise")                           
                           )
