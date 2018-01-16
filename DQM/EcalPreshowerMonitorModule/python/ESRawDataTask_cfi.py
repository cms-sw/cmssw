import FWCore.ParameterSet.Config as cms
    
ecalPreshowerRawDataTask = DQMStep1Module('ESRawDataTask',
                                          prefixME = cms.untracked.string('EcalPreshower'),
                                          FEDRawDataCollection = cms.InputTag("rawDataCollector"),
                                          ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                          OutputFile = cms.untracked.string("")
                                          )

