import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoEcalFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ecalRecHit_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_islandBasicClusters_*_*', 
        'keep *_islandSuperClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'keep *_correctedIslandBarrelSuperClusters_*_*', 
        'keep *_correctedIslandEndcapSuperClusters_*_*', 
        'keep *_correctedHybridSuperClusters_*_*', 
        'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep *_preshowerClusterShape_*_*')
)
# RECO content
RecoEcalRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_islandBasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*')
)
# AOD content
RecoEcalAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_islandBasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep recoSuperClusters_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*')
)

