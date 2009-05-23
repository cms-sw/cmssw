import FWCore.ParameterSet.Config as cms
    
ecalPreshowerIntegrityTask = cms.EDAnalyzer('ESIntegrityTask',
                                            prefixME = cms.untracked.string('EcalPreshower'),
                                            ESDCCCollections = cms.InputTag("esRawToDigi"),
                                            ESKChipCollections = cms.InputTag("esRawToDigi"),
                                            OutputFile = cms.untracked.string("")
                                            )

