import FWCore.ParameterSet.Config as cms

hiCentrality = cms.EDFilter("reco::CentralityProducer",

                              doFilter = cms.bool(False),

                              produceHFhits = cms.bool(True),
                              produceHFtowers = cms.bool(True),
                              produceEcalhits = cms.bool(False),
                              produceBasicClusters = cms.bool(True),
                              produceZDChits = cms.bool(False),

                              srcHFhits = cms.InputTag("hfreco"),
                              srcTowers = cms.InputTag("towerMaker"),
                              srcEBhits = cms.InputTag("EcalRecHitsEB"),
                              srcEEhits = cms.InputTag("EcalRecHitsEE"),
                              srcBasicClustersEB = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
                              srcBasicClustersEE = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
                              srcZDChits = cms.InputTag("")

                              )


