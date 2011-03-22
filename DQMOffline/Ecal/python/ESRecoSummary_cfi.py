import FWCore.ParameterSet.Config as cms

ecalPreshowerRecoSummary = cms.EDAnalyzer("ESRecoSummary",
    prefixME = cms.untracked.string('EcalPreshower'),    
    superClusterCollection_EE = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    recHitCollection_ES       = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    ClusterCollectionX_ES     = cms.InputTag("multi5x5SuperClustersWithPreshower","preshowerXClusters"),
    ClusterCollectionY_ES     = cms.InputTag("multi5x5SuperClustersWithPreshower","preshowerYClusters"),
)

