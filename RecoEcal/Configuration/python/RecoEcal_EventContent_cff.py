import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoEcalFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_reducedEcalRecHits*_*_*', 
        'keep *_interestingEcalDetId*_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_islandBasicClusters_*_*', 
        'keep *_islandSuperClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'keep *_correctedMulti5x5*_*_*', 
        'keep *_multi5x5*_*_*', 
        'keep *_correctedIslandBarrelSuperClusters_*_*', 
        'keep *_correctedIslandEndcapSuperClusters_*_*', 
        'keep *_correctedHybridSuperClusters_*_*', 
        'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep *_preshowerClusterShape_*_*')
)
# RECO content
RecoEcalRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_islandBasicClusters_*_*', 
        'keep *_multi5x5BasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'keep EcalRecHitsSorted_reducedEcalRecHits*_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'drop recoClusterShapes_*_*_*', 
        'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep *_correctedMulti5x5SuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 
        'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*')
)
# AOD content
RecoEcalAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_islandBasicClusters_*_*', 
        'keep *_multi5x5BasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'keep EcalRecHitsSorted_reducedEcalRecHits*_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'drop recoClusterShapes_*_*_*', 
        'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep recoSuperClusters_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*')
)

