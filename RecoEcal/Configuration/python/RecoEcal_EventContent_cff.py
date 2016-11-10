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
        # Particle Flow superclusters (only SuperCluster and CaloCluster outputs, not association map from PFClusters)
        'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*'
        )
)

_phase2_hgcal_scCommands = ['keep *_particleFlowSuperClusterHGCal_*_*']
_phase2_hgcal_scCommandsAOD = ['keep recoSuperClusters_particleFlowSuperClusterHGCal__*',
                               'keep recoCaloClusters_particleFlowSuperClusterHGCal__*']
_phase2_hgcal_RecoEcalFEVT = RecoEcalFEVT.clone()
_phase2_hgcal_RecoEcalFEVT.outputCommands += _phase2_hgcal_scCommands
_phase2_hgcal_RecoEcalRECO = RecoEcalRECO.clone()
_phase2_hgcal_RecoEcalRECO.outputCommands += _phase2_hgcal_scCommands
_phase2_hgcal_RecoEcalAOD  = RecoEcalAOD.clone()
_phase2_hgcal_RecoEcalAOD.outputCommands += _phase2_hgcal_scCommandsAOD
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith( RecoEcalFEVT, _phase2_hgcal_RecoEcalFEVT )
phase2_hgcal.toReplaceWith( RecoEcalRECO, _phase2_hgcal_RecoEcalRECO )
phase2_hgcal.toReplaceWith( RecoEcalAOD , _phase2_hgcal_RecoEcalAOD  )

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
#HI-specific products needed in pp scenario special configurations
for e in [pA_2016, peripheralPbPb]:
    for ec in [RecoEcalRECO.outputCommands, RecoEcalFEVT.outputCommands]:
        e.toModify( ec, func=lambda outputCommands: outputCommands.extend(['keep recoCaloClusters_islandBasicClusters_*_*'])
                    )
