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
        'keep recoPFCandidates_particleFlowTmp_*_REPROD', 
        'keep recoCandidatesOwned_*_*_REPROD', 
        'keep recoPFSimParticles_*_*_REPROD', 
        'keep recoTracks_*_*_*',
        'keep recoTrackExtras_*_*_*', 
        'keep *_electronGsfTracks_*_*', 
        'keep *_electronMergedSeeds_*_*', 
        'keep recoCaloJets_*_*_*', 
        'keep recoPFJets_*_*_REPROD', 
        'keep recoGenJets_*_*_REPROD', 
        'keep recoCaloMETs_*_*_*', 
        'keep recoPFMETs_*_*_REPROD', 
        'keep recoMETs_tcMet_*_*', 
        'keep recoGenParticles_*_*_REPROD', 
        'keep recoGenParticlesRefs_*_*_REPROD', 
        'keep CaloTowersSorted_towerMaker_*_*', 
        'keep *_offlinePrimaryVertices_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep edmHepMCProduct_*_*_*',
        'keep recoMuons_*_*_*',
        'keep *_photons_*_*',
        'keep recoConversions_*_*_*',
        'keep recoPFConversions_*_*_REPROD',
	'keep recoNuclearInteractions_*_*_*',
        'keep *_pfNuclear_*_REPROD',
	'keep *_generalV0Candidates_*_*',
        'keep recoPFV0s_*_*_REPROD',                      
        'keep recoPFDisplacedVertexs_*_*_REPROD',                      
        'keep recoPFDisplacedTrackerVertexs_*_*_REPROD',
        'keep *_correctedHybridSuperClusters_*_*',
        'keep *_correctedMulti5x5SuperClustersWithPreshower_*_*',
        'keep recoPhotons_photons_*_*',
        'keep recoPhotonCores_photonCore_*_*',
        'keep recoGsfElectrons_electronsCiCLoose_*_*',
        'keep recoGsfElectrons_mvaElectrons_*_*',
        'keep recoGsfElectronCores_ecalDrivenGsfElectronCores_*_*'                                           
   )
)


