import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoEcalFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        #selected digis
        'keep *_selectDigi_*_*',
	# Hits
	'keep *_reducedEcalRecHits*_*_*', 
        'keep *_interestingEcalDetId*_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
	# Barrel clusters
        'keep *_hybridSuperClusters_*_*',
        'keep *_uncleanedHybridSuperClusters_*_*',
        'keep *_correctedHybridSuperClusters_*_*',
	# Endcap clusters
        'keep *_multi5x5*_*_*',
        'keep *_correctedMulti5x5*_*_*',
        # Preshower clusters
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*',
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
	'keep EcalRecHitsSorted_reducedEcalRecHits*_*_*',
	# Barrel clusters
        'keep *_hybridSuperClusters_*_*',
        'keep *_uncleanedHybridSuperClusters_*_*',
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
	# Endcap clusters
        'keep *_multi5x5BasicClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
	# Preshower clusters
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*',
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
        'keep EcalRecHitsSorted_reducedEcalRecHits*_*_*', 
	# Barrel clusters
        'keep recoCaloClusters_hybridSuperClusters_*_*',
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
	# Endcap clusters	
        'keep recoCaloClusters_multi5x5BasicClusters_multi5x5EndcapBasicClusters_*', 
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
	# Preshower clusters
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*',
        )
)
