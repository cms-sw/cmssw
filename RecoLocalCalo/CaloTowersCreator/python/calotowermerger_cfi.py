import FWCore.ParameterSet.Config as cms
# the extraTowerTag should point to a collection of towers reconstructed
# from rejected hits
calotowermerger = cms.EDProducer("CaloTowersMerger",
                                 regularTowerTag=cms.InputTag('towerMaker'),
                                 extraTowerTag=cms.InputTag('')
                                 )
                                     
