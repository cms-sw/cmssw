# The following comments couldn't be translated into the new config version:

#        "keep recoPFClusters_*_*_*",
#        "keep recoPFBlocks_*_*_*",	

import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoParticleFlowFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*',
        'keep recoTracks_firstnuclearWithMaterialTracks_*_*', 
        'keep recoTrackExtras_firstnuclearWithMaterialTracks_*_*', 
        'keep TrackingRecHitsOwned_firstnuclearWithMaterialTracks_*_*',        
        'keep *_firstnuclearInteractionMaker_*_*',                                 
        'keep recoPFClusters_*_*_*', 
        'keep recoPFBlocks_*_*_*', 
        'keep recoPFCandidates_*_*_*')
)
# RECO content
RecoParticleFlowRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*',
        'keep recoTracks_firstnuclearWithMaterialTracks_*_*', 
        'keep recoTrackExtras_firstnuclearWithMaterialTracks_*_*', 
        'keep TrackingRecHitsOwned_firstnuclearWithMaterialTracks_*_*', 
        'keep *_firstnuclearInteractionMaker_*_*',                                           
        'keep recoPFClusters_*_*_*', 
        'keep recoPFBlocks_*_*_*', 
        'keep recoPFCandidates_*_*_*')
)
# AOD content
RecoParticleFlowAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*',
        'keep recoTracks_firstnuclearWithMaterialTracks_*_*',                                    
        'keep *_firstnuclearInteractionMaker_*_*',
        'keep recoPFCandidates_*_*_*')
)

