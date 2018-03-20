import FWCore.ParameterSet.Config as cms
    
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerTrendTask = DQMEDAnalyzer('ESTrendTask',
                                        RecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                                        ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                        prefixME = cms.untracked.string("EcalPreshower") 
                                        )

