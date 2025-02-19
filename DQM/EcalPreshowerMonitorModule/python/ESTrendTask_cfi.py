import FWCore.ParameterSet.Config as cms
    
ecalPreshowerTrendTask = cms.EDAnalyzer('ESTrendTask',
                                        RecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                                        ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                        prefixME = cms.untracked.string("EcalPreshower") 
                                        )

