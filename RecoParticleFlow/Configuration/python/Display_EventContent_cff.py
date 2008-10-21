# The following comments couldn't be translated into the new config version:

#Tracks without extra and hits

import FWCore.ParameterSet.Config as cms

DisplayEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoPFRecHits_*_*_*', 
        'keep recoPFClusters_*_*_*', 
        'keep recoPFRecTracks_*_*_*', 
        'keep recoGsfPFRecTracks_*_*_*', 
        'keep recoPFBlocks_particleFlowBlock_*_*', 
        'keep recoPFCandidates_particleFlow_*_*', 
        'keep recoCandidatesOwned_*_*_*', 
        'keep recoPFSimParticles_*_*_*', 
        'keep recoTracks_*_*_*', 
        'keep recoCaloJets_*_*_*', 
        'keep recoPFJets_*_*_*', 
        'keep recoGenParticles_*_*_*', 
        'keep recoGenParticlesRefs_*_*_*', 
        'keep CaloTowersSorted_*_*_*', 
        'keep *_offlinePrimaryVertices_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep edmHepMCProduct_*_*_*')
)


