import FWCore.ParameterSet.Config as cms
    
ecalPreshowerRawDataTask = cms.EDAnalyzer('ESRawDataTask',
                                          prefixME = cms.untracked.string('EcalPreshower'),
                                          FEDRawDataCollection = cms.InputTag("source"),
                                          ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                          OutputFile = cms.untracked.string("")
                                          )

