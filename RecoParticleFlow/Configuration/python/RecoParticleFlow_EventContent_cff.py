# The following comments couldn't be translated into the new config version:

#        "keep recoPFClusters_*_*_*",
#        "keep recoPFBlocks_*_*_*",	

import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoParticleFlowFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*')
)
# RECO content
RecoParticleFlowRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*')
)
# AOD content
RecoParticleFlowAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFCandidates_*_*_*', 'keep recoTracks_secStep_*_*', 'keep recoTracks_thStep_*_*')
)

