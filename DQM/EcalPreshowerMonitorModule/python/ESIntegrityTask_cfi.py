import FWCore.ParameterSet.Config as cms
    
ecalPreshowerIntegrityTask = cms.EDAnalyzer('ESIntegrityTask',
                                            ESDCCCollections = cms.InputTag("esRawToDigi"),
                                            ESKChipCollections = cms.InputTag("esRawToDigi"),
                                            OutputFile = cms.untracked.string("")
                                            )

