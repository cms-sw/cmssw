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
    'keep recoPFRecHits_particleFlowClusterHFEM_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHFHAD_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
    #'keep recoPFClusters_*_*_*',
    'keep recoPFClusters_particleFlowClusterECAL_*',
    'keep recoPFClusters_particleFlowClusterHCAL_*',
    'keep recoPFClusters_particleFlowClusterHFEM_*',
    'keep recoPFClusters_particleFlowClusterHFHAD_*',
    'keep recoPFClusters_particleFlowClusterPS_*',
    #'keep recoPFBlocks_*_*_*',
    'keep recoPFBlocks_particleFlowBlock_*',
    #'keep recoPFCandidates_*_*_*',
    'keep recoPFCandidates_particleFlow_*',
    #'keep recoPFDisplacedVertexs_*_*_*',
    'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*',
    'keep *_pfElectronTranslator_*_*',
    'keep *_trackerDrivenElectronSeeds_preid_*')
    )
# RECO content
RecoParticleFlowRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'drop CaloTowersSorted_towerMakerPF_*_*', 
    #'keep recoPFRecHits_*_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHFEM_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHFHAD_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
    #'keep recoPFClusters_*_*_*',
    'keep recoPFClusters_particleFlowClusterECAL_*',
    'keep recoPFClusters_particleFlowClusterHCAL_*',
    'keep recoPFClusters_particleFlowClusterHFEM_*',
    'keep recoPFClusters_particleFlowClusterHFHAD_*',
    'keep recoPFClusters_particleFlowClusterPS_*',
    #'keep recoPFBlocks_*_*_*',
    'keep recoPFBlocks_particleFlowBlock_*',
    #'keep recoPFCandidates_*_*_*',
    'keep recoPFCandidates_particleFlow_*',
    #'keep recoPFDisplacedVertexs_*_*_*',
    'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*',
    'keep *_pfElectronTranslator_*_*',
    'keep *_trackerDrivenElectronSeeds_preid_*')
)    
    
# AOD content
RecoParticleFlowAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'drop CaloTowersSorted_towerMakerPF_*_*',
    'drop *_pfElectronTranslator_*_*',
    #'keep recoPFRecHits_*_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHFEM_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterHFHAD_Cleaned_*',
    'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitHCAL_Cleaned_*',
    'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
    #'keep recoPFCandidates_*_*_*',
    'keep recoPFCandidates_particleFlow_*',
    'keep recoCaloClusters_pfElectronTranslator_*_*',
    'keep recoPreshowerClusters_pfElectronTranslator_*_*',
    'keep recoSuperClusters_pfElectronTranslator_*_*')
)
