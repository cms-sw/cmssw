import FWCore.ParameterSet.Config as cms
    
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerRawDataTask = DQMEDAnalyzer('ESRawDataTask',
                                          prefixME = cms.untracked.string('EcalPreshower'),
                                          FEDRawDataCollection = cms.InputTag("rawDataCollector"),
                                          ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                          OutputFile = cms.untracked.string("")
                                          )

