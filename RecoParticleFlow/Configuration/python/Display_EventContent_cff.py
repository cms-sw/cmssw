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
        'keep recoPFCandidates_particleFlowTmp_*_*',                                            
        'keep recoCandidatesOwned_*_*_*', 
        'keep recoPFSimParticles_*_*_*', 
        'keep recoTracks_*_*_*',
        'keep recoTrackExtras_*_*_*',                                    
        'keep recoGsfTracks_*_*_*', 
        'keep recoCaloJets_*_*_*', 
        'keep recoPFJets_*_*_*', 
        'keep recoCaloMETs_*_*_*', 
        'keep recoPFMETs_*_*_*', 
        'keep recoMETs_tcMet_*_*', 
        'keep recoGenParticles_*_*_*', 
        'keep recoGenParticlesRefs_*_*_*', 
        'keep CaloTowersSorted_towerMaker_*_*', 
        'keep *_offlinePrimaryVertices_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep edmHepMCProduct_*_*_*',
        'keep recoConversions_*_*_*',
        'keep recoPFConversions_*_*_*',
        'keep recoMuons_*_*_*',
        'keep recoNuclearInteractions_*_*_*',
        'keep *_pfNuclear_*_*',
	'keep *_generalV0Candidates_*_*',
        'keep recoPFV0s_*_*_*',                      
        'keep recoPFDisplacedVertexs_*_*_*',                      
        'keep recoPFDisplacedTrackerVertexs_*_*_*',                               
        'keep *_correctedHybridSuperClusters_*_*',
        'keep *_hybridSuperClusters_hybridBarrelBasicClusters_*',                                            
        'keep *_correctedMulti5x5SuperClustersWithPreshower_*_*',
        'keep *_multi5x5SuperClusters_multi5x5EndcapBasicClusters_*',                                           
        'keep recoGsfTrackExtras_electronGsfTracks_*_*',
        'keep recoTrackExtras_electronGSGsfTrackCandidates_*_*',
        'keep recoTrackExtras_electronGsfTracks_*_*',         
        'keep recoElectronSeeds_electronMergedSeeds_*_*',
        'keep recoPhotons_photons_*_*',
        'keep recoPhotonCores_photonCore_*_*',
        'keep recoPFCandidateElectronExtras_particleFlowTmp_*_*',
        'keep recoPFCandidatePhotonExtras_particleFlowTmp_*_*',
        'keep recoGsfElectrons_electronsCiCLoose_*_*',
        'keep recoGsfElectrons_mvaElectrons_*_*',                                   
        'keep recoGsfElectronCores_ecalDrivenGsfElectronCores_*_*'                                           
   )
)


