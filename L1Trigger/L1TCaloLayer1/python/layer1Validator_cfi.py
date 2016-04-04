import FWCore.ParameterSet.Config as cms

layer1Validator = cms.EDAnalyzer('L1TCaloLayer1Validator',
                                 testSource = cms.InputTag("l1tCaloLayer1SpyDigis"),
                                 emulSource = cms.InputTag("layer1EmulatorDigis"),
                                 verbose = cms.bool(False)
                                 )
