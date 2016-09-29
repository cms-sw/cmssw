# The following comments couldn't be translated into the new config version:

#        "keep recoPFClusters_*_*_*",
#        "keep recoPFBlocks_*_*_*",	

import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoParticleFlowFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'drop CaloTowersSorted_towerMakerPF_*_*',
    #'keep recoPFRecHits_*_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHF_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHBHE_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHF_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
    #'keep recoPFClusters_*_*_*',
    'keep recoPFClusters_particleFlowClusterECAL_*_*',    
    'keep recoPFClusters_particleFlowClusterHCAL_*_*',
    'keep recoPFClusters_particleFlowClusterHO_*_*',
    'keep recoPFClusters_particleFlowClusterHF_*_*',
    'keep recoPFClusters_particleFlowClusterPS_*_*',
    #'keep recoPFBlocks_*_*_*',
    'keep recoPFBlocks_particleFlowBlock_*_*',
    #'keep recoPFCandidates_*_*_*',
    'keep recoPFCandidates_particleFlowEGamma_*_*',
    'keep recoCaloClusters_particleFlowEGamma_*_*',
    'keep recoSuperClusters_particleFlowEGamma_*_*',    
    'keep recoConversions_particleFlowEGamma_*_*',
    'keep recoPFCandidates_particleFlow_*_*',
    'keep recoPFCandidates_particleFlowTmp_*_*',
    'drop recoPFCandidates_particleFlowTmp__*',
    #'keep recoPFDisplacedVertexs_*_*_*',
    'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*_*',
    'keep *_pfElectronTranslator_*_*',
    'keep *_pfPhotonTranslator_*_*',
    'keep *_particleFlow_electrons_*',
    'keep *_particleFlow_photons_*',
    'keep *_trackerDrivenElectronSeeds_preid_*',
    'keep *_particleFlowPtrs_*_*',
    'keep *_particleFlowTmpPtrs_*_*'
        )
    )
# RECO content
RecoParticleFlowRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'drop CaloTowersSorted_towerMakerPF_*_*', 
    #'keep recoPFRecHits_*_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHF_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHBHE_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHF_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
    #'keep recoPFClusters_*_*_*',
    'keep recoPFClusters_particleFlowClusterECAL_*_*',
    'keep recoPFClusters_particleFlowClusterHCAL_*_*',
    'keep recoPFClusters_particleFlowClusterHO_*_*',
    'keep recoPFClusters_particleFlowClusterHF_*_*',
    'keep recoPFClusters_particleFlowClusterPS_*_*',
    #'keep recoPFBlocks_*_*_*',
    'keep recoPFBlocks_particleFlowBlock_*_*',
    #'keep recoPFCandidates_*_*_*',
    'keep recoPFCandidates_particleFlowEGamma_*_*',
    'keep recoCaloClusters_particleFlowEGamma_*_*',
    'keep recoSuperClusters_particleFlowEGamma_*_*',
    'keep recoConversions_particleFlowEGamma_*_*',
    'keep recoPFCandidates_particleFlow_*_*',
    'keep recoPFCandidates_particleFlowTmp_electrons_*',
    'keep recoPFCandidates_particleFlowTmp_*_*',
    'drop recoPFCandidates_particleFlowTmp__*',
    #'keep recoPFDisplacedVertexs_*_*_*',
    'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*_*',
    'keep *_pfElectronTranslator_*_*',
    'keep *_pfPhotonTranslator_*_*',
    'keep *_particleFlow_electrons_*',
    'keep *_particleFlow_photons_*',
    'keep *_particleFlow_muons_*',
    'keep *_trackerDrivenElectronSeeds_preid_*',
    'keep *_particleFlowPtrs_*_*',
    'keep *_particleFlowTmpPtrs_*_*'
        )
)    
    
# AOD content
RecoParticleFlowAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'drop CaloTowersSorted_towerMakerPF_*_*',
    'drop *_pfElectronTranslator_*_*',
    #'keep recoPFRecHits_*_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHF_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHBHE_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHF_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
    #'keep recoPFCandidates_*_*_*',
    'keep recoCaloClusters_particleFlowEGamma_*_*',
    'keep recoSuperClusters_particleFlowEGamma_*_*',
    'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
    'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
    'keep recoConversions_particleFlowEGamma_*_*',
    'keep recoPFCandidates_particleFlow_*_*',
    'keep recoPFCandidates_particleFlowTmp_*_*',
    'drop recoPFCandidates_particleFlowTmp__*',
    'keep *_particleFlow_electrons_*',
    'keep *_particleFlow_photons_*',
    'keep *_particleFlow_muons_*',
    'keep recoCaloClusters_pfElectronTranslator_*_*',
    'keep recoPreshowerClusters_pfElectronTranslator_*_*',
    'keep recoSuperClusters_pfElectronTranslator_*_*',
    'keep recoCaloClusters_pfPhotonTranslator_*_*',
    'keep recoPreshowerClusters_pfPhotonTranslator_*_*',
    'keep recoSuperClusters_pfPhotonTranslator_*_*',
    'keep recoPhotons_pfPhotonTranslator_*_*',
    'keep recoPhotonCores_pfPhotonTranslator_*_*',
    'keep recoConversions_pfPhotonTranslator_*_*',
    'keep *_particleFlowPtrs_*_*',
    'keep *_particleFlowTmpPtrs_*_*'
        )
)

def _modifyPFEventContentForHGCalRECO( obj ):
    obj.outputCommands.append('keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*')

def _modifyPFEventContentForHGCalFEVT( obj ):
    obj.outputCommands.append('keep recoPFRecHits_particleFlowClusterECAL__*')
    obj.outputCommands.append('keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*')


# mods for HGCAL
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( RecoParticleFlowFEVT, outputCommands = RecoParticleFlowFEVT.outputCommands + [ 
        'keep recoPFRecHits_particleFlowClusterECAL__*',
        'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*',
        'keep recoPFRecHits_particleFlowRecHitHGC__*',
        'keep recoPFRecHits_particleFlowRecHitHGC_Cleaned_*',
        'keep recoPFClusters_particleFlowClusterHGCal__*',
        'keep *_simPFProducer_*_*',
        'keep *_particleFlowTmpBarrel_*_*',
    ]
)
phase2_hgcal.toModify( RecoParticleFlowRECO, outputCommands = RecoParticleFlowRECO.outputCommands + [ 'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHGC_Cleaned_*', 'keep recoPFClusters_particleFlowClusterHGCal__*', 'keep recoPFBlocks_simPFProducer_*_*', 'keep recoSuperClusters_simPFProducer_*_*','keep *_particleFlowTmpBarrel_*_*' ] )
phase2_hgcal.toModify( RecoParticleFlowAOD,  outputCommands = RecoParticleFlowAOD.outputCommands + [ 'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHGC_Cleaned_*', 'keep recoPFClusters_particleFlowClusterHGCal__*', 'keep recoSuperClusters_simPFProducer_*_*' ] )

#timing
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( 
    RecoParticleFlowFEVT, 
    outputCommands = RecoParticleFlowFEVT.outputCommands + [
        'keep *_ecalBarrelClusterFastTimer_*_*'
        ])
phase2_timing.toModify( 
    RecoParticleFlowRECO, 
    outputCommands = RecoParticleFlowRECO.outputCommands + [
        'keep *_ecalBarrelClusterFastTimer_*_*'
        ])
phase2_timing.toModify( 
    RecoParticleFlowAOD, 
    outputCommands = RecoParticleFlowAOD.outputCommands + [
        'keep *_ecalBarrelClusterFastTimer_*_*'
        ])
