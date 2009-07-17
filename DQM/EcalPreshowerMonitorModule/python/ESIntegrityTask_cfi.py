import FWCore.ParameterSet.Config as cms
    
ecalPreshowerIntegrityTask = cms.EDAnalyzer('ESIntegrityTask',
                                            prefixME = cms.untracked.string('EcalPreshower'),
                                            ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                            ESKChipCollections = cms.InputTag("ecalPreshowerDigis"),
                                            OutputFile = cms.untracked.string("")
                                            )

