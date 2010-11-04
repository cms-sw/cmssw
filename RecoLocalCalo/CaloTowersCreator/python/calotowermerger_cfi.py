import FWCore.ParameterSet.Config as cms
calotowermerger = cms.EDProducer("CaloTowersMerger",
                                 towerTag1=cms.InputTag('towerMaker'),
                                 towerTag2=cms.InputTag('')
                                 )
                                     
