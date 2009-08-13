# The following comments couldn't be translated into the new config version:

#Tracks without extra and hits

import FWCore.ParameterSet.Config as cms

DisplayEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoPFRecHits_*_*_REPROD', 
        'keep recoPFClusters_*_*_REPROD', 
        'keep recoPFRecTracks_*_*_REPROD', 
        'keep recoGsfPFRecTracks_*_*_REPROD', 
        'keep recoPFBlocks_particleFlowBlock_*_REPROD', 
        'keep recoPFCandidates_particleFlow_*_REPROD', 
        'keep recoCandidatesOwned_*_*_REPROD', 
        'keep recoPFSimParticles_*_*_REPROD', 
        'keep recoTracks_*_*_REPROD',
        'keep recoGsfTracks_*_*_REPROD', 
        'keep recoCaloJets_*_*_REPROD', 
        'keep recoPFJets_*_*_REPROD', 
        'keep recoGenJets_*_*_REPROD', 
        'keep recoCaloMETs_*_*_REPROD', 
        'keep recoPFMETs_*_*_REPROD', 
        'keep recoMETs_tcMet_*_REPROD', 
        'keep recoGenParticles_*_*_REPROD', 
        'keep recoGenParticlesRefs_*_*_REPROD', 
        'keep CaloTowersSorted_towerMaker_*_REPROD', 
        'keep *_offlinePrimaryVertices_*_REPROD', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_REPROD', 
        'keep edmHepMCProduct_*_*_REPROD',
        'keep recoMuons_*_*_REPROD',
	'keep *_generalV0Candidates_*_REPROD'                                          
   )
)


