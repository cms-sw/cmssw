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
        'keep recoTracks_*_*_REPROD',
        'keep recoTrackExtras_*_*_REPROD',                                    
        'keep *_electronGsfTracks_*_REPROD', 
        'keep *_electronMergedSeeds_*_REPROD', 
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
        'keep *_photons_*_REPROD',
        'keep recoConversions_*_*_REPROD',
        'keep recoPFConversions_*_*_REPROD',
	'keep recoNuclearInteractions_*_*_REPROD',
        'keep *_pfNuclear_*_REPROD',
	'keep *_generalV0Candidates_*_REPROD',
        'keep recoPFV0s_*_*_REPROD',                      
        'keep recoPFDisplacedVertexs_*_*_REPROD',                      
        'keep recoPFDisplacedTrackerVertexs_*_*_REPROD',
        'keep *_correctedHybridSuperClusters_*_*',
        'keep *_hybridSuperClusters_hybridBarrelBasicClusters_*',                                            
        'keep *_correctedMulti5x5SuperClustersWithPreshower_*_*',
        'keep *_multi5x5SuperClusters_multi5x5EndcapBasicClusters_*',                                           
        'keep recoGsfTrackExtras_electronGsfTracks_*_REPROD',
        'keep recoTrackExtras_electronGSGsfTrackCandidates_*_REPROD',
        'keep recoTrackExtras_electronGsfTracks_*_REPROD',         
        'keep recoElectronSeeds_electronMergedSeeds_*_REPROD',
        'keep recoPhotons_photons_*_REPROD',
        'keep recoPhotonCores_photonCore_*_REPROD',
        'keep recoPFCandidateElectronExtras_particleFlowTmp_*_REPROD',
        'keep recoPFCandidatePhotonExtras_particleFlowTmp_*_REPROD',
        'keep recoGsfElectrons_electronsCiCLoose_*_*',
        'keep recoGsfElectrons_mvaElectrons_*_*',
        'keep recoGsfElectronCores_ecalDrivenGsfElectronCores_*_*'                                           
   )
)


