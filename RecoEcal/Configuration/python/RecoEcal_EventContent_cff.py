import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoEcalFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        #selected digis
        'keep *_selectDigi_*_*',
	# Hits
	'keep *_reducedEcalRecHitsEB_*_*',
        'keep *_reducedEcalRecHitsEE_*_*',
        'keep *_reducedEcalRecHitsES_*_*', 
        'keep *_interestingEcalDetId*_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
	# Barrel clusters
        'keep *_hybridSuperClusters_*_*',
        'keep *_correctedHybridSuperClusters_*_*',
	# Endcap clusters
        'keep *_multi5x5*_*_*',
        'keep *_correctedMulti5x5*_*_*',
        # Preshower clusters
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*',
        # Particle Flow superclusters
        'keep *_particleFlowSuperClusterECAL_*_*',
	# DROP statements
	'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*',
        'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*')
)
# RECO content
RecoEcalRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        #selected digis
        'keep *_selectDigi_*_*',
	# Hits
	'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsES_*_*',
	# Barrel clusters
        'keep *_hybridSuperClusters_*_*',
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
	# Endcap clusters
        'keep *_multi5x5SuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
	# Preshower clusters
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*',
        # Particle Flow superclusters
        'keep *_particleFlowSuperClusterECAL_*_*',
	# DROP statements
        'drop recoClusterShapes_*_*_*', 
        'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*',
        'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*',
        'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*')
)
# AOD content
RecoEcalAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        #selected digis
        'keep *_selectDigi_*_*',
	# Hits
        'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsES_*_*',
	# Barrel clusters (uncleaned only in separate collections)
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
        'keep recoCaloClusters_hybridSuperClusters_*_*',
        'keep recoSuperClusters_hybridSuperClusters_uncleanOnlyHybridSuperClusters_*',
	# Endcap clusters	
	'keep recoCaloClusters_multi5x5SuperClusters_multi5x5EndcapBasicClusters_*',
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
	# Preshower clusters
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*',
        # Particle Flow superclusters
        'keep *_particleFlowSuperClusterECAL_*_*'
        )
)
