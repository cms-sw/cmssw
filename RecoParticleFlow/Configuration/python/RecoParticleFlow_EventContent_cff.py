# The following comments couldn't be translated into the new config version:

#        "keep recoPFClusters_*_*_*",
#        "keep recoPFBlocks_*_*_*",	

import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoParticleFlowFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*', 
        'keep recoPFClusters_*_*_*',
        'keep recoPFBlocks_*_*_*', 
        'keep recoPFCandidates_*_*_*',
        'keep *_pfElectronTranslator_*_*')
)
# RECO content
RecoParticleFlowRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*', 
        'keep recoPFClusters_*_*_*', 
        'keep recoPFBlocks_*_*_*',
        'keep recoPFCandidates_*_*_*',
        'keep *_pfElectronTranslator_*_*')
)
# AOD content
RecoParticleFlowAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*',
        'drop *_pfElectronTranslator_*_*',
        'keep recoPFCandidates_*_*_*',
        'keep recoCaloClusters_pfElectronTranslator_*_*',
        'keep recoPreshowerClusters_pfElectronTranslator_*_*',
        'keep recoSuperClusters_pfElectronTranslator_*_*')
)
