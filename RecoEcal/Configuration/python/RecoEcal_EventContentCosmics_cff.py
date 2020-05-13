import FWCore.ParameterSet.Config as cms

# AOD content
RecoEcalAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'keep *_islandBasicClusters_*_*', 
        'keep *_fixedMatrixBasicClusters_*_*', 
        'keep *_cosmicBasicClusters_*_*', 
        'keep *_cosmicSuperClusters_*_*', 
        'keep recoCaloClusters_hybridSuperClusters_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep recoSuperClusters_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusters_fixedMatrixSuperClustersWithPreshower_*_*', 
        'keep recoSuperClusters_correctedFixedMatrixSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 
        'keep recoPreshowerClusterShapes_fixedMatrixPreshowerClusterShape_*_*')
)

# RECO content
RecoEcalRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_uncleanedHybridSuperClusters_*_*',                         
        'keep *_correctedFixedMatrixSuperClustersWithPreshower_*_*',
        'keep *_correctedEndcapSuperClustersWithPreshower_*_*')
)
RecoEcalRECO.outputCommands.extend(RecoEcalAOD.outputCommands)

RecoEcalFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'keep *_ecalRecHit_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_islandSuperClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'keep *_correctedFixedMatrix*_*_*', 
        'keep *_fixedMatrix*_*_*', 
        'keep *_correctedIslandBarrelSuperClusters_*_*', 
        'keep *_correctedIslandEndcapSuperClusters_*_*', 
        'keep *_correctedHybridSuperClusters_*_*', 
        'keep *_preshowerClusterShape_*_*')
)
RecoEcalFEVT.outputCommands.extend(RecoEcalRECO.outputCommands)
