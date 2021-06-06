import FWCore.ParameterSet.Config as cms

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
        'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterOOTECAL_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterOOTECAL_*_*')
)
_phase2_hgcal_scCommandsAOD = ['keep recoSuperClusters_particleFlowSuperClusterHGCal__*',
                               'keep recoCaloClusters_particleFlowSuperClusterHGCal__*',
                               'keep recoSuperClusters_particleFlowSuperClusterHGCalFromMultiCl__*',
                               'keep recoCaloClusters_particleFlowSuperClusterHGCalFromMultiCl__*']

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(RecoEcalAOD,
    outputCommands = RecoEcalAOD.outputCommands + _phase2_hgcal_scCommandsAOD)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
#HI-specific products needed in pp scenario special configurations
for e in [pA_2016, peripheralPbPb, pp_on_AA, pp_on_XeXe_2017, ppRef_2017]:
    e.toModify( RecoEcalAOD.outputCommands, 
                func=lambda outputCommands: outputCommands.extend(['keep recoSuperClusters_correctedIslandBarrelSuperClusters_*_*',
                                                                   'keep recoSuperClusters_correctedIslandEndcapSuperClusters_*_*'])
              )

# RECO content
RecoEcalRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
	# Barrel clusters
        'keep *_hybridSuperClusters_*_*',
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
	# Endcap clusters
        'keep *_multi5x5SuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*',
        # Particle Flow superclusters
        'keep *_particleFlowSuperClusterECAL_*_*',
        'keep *_particleFlowSuperClusterOOTECAL_*_*',
	# DROP statements 
        'drop recoClusterShapes_*_*_*', 
        'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*',
        'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*',
        'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*')
)
RecoEcalRECO.outputCommands.extend(RecoEcalAOD.outputCommands)
_phase2_hgcal_scCommands = ['keep *_particleFlowSuperClusterHGCal_*_*',
                            'keep *_particleFlowSuperClusterHGCalFromMultiCl_*_*']
phase2_hgcal.toModify(RecoEcalRECO,
    outputCommands = RecoEcalRECO.outputCommands + _phase2_hgcal_scCommands)

for e in [pA_2016, peripheralPbPb, pp_on_AA, pp_on_XeXe_2017, ppRef_2017]:
    e.toModify( RecoEcalRECO.outputCommands,
                func=lambda outputCommands: outputCommands.extend(['keep recoCaloClusters_islandBasicClusters_*_*'])
              )

# Full Event content 
RecoEcalFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
	# Hits
	'keep *_reducedEcalRecHitsEB_*_*',
        'keep *_reducedEcalRecHitsEE_*_*',
        'keep *_reducedEcalRecHitsES_*_*', 
        'keep *_interestingEcalDetId*_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
	# Barrel clusters
        'keep *_correctedHybridSuperClusters_*_*',
	# Endcap clusters
        'keep *_multi5x5*_*_*')
)
RecoEcalFEVT.outputCommands.extend(RecoEcalRECO.outputCommands)
