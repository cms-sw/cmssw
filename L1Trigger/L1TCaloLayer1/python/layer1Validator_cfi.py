import FWCore.ParameterSet.Config as cms

layer1Validator = cms.EDAnalyzer('L1TCaloLayer1Validator',
                                 testTowerToken = cms.InputTag("l1tCaloLayer1SpyDigis"),
                                 emulTowerToken = cms.InputTag("layer1EmulatorDigis"),
                                 testRegionToken = cms.InputTag("l1tCaloLayer1SpyDigis"),
                                 emulRegionToken = cms.InputTag("layer1EmulatorDigis"),
                                 validateTowers = cms.bool(True),
                                 validateRegions = cms.bool(True),
                                 verbose = cms.bool(False)
                                 )
