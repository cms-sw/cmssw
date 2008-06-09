import FWCore.ParameterSet.Config as cms

zToEEEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoCandidatesOwned_allElectrons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allSuperClusters_*_*', 
        'keep *_allElectronIsolations_*_*', 
        'keep *_allTrackIsolations_*_*', 
        'keep *_allSuperClusterIsolations_*_*', 
        'keep *_zToEE_*_*', 
        'keep *_zToEEOneTrack_*_*', 
        'keep *_zToEEOneSuperCluster_*_*', 
        'keep *_allElectronsGenParticlesMatch_*_*', 
        'keep *_allTracksGenParticlesLeptonMatch_*_*', 
        'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 
        'keep *_zToEEGenParticlesMatch_*_*', 
        'keep *_zToEEOneTrackGenParticlesMatch_*_*', 
        'keep *_zToEEOneSuperClusterGenParticlesMatch_*_*')
)
zToEEEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToEEPath', 
            'zToEEOneTrackPath', 
            'zToEEOneSuperClusterPath')
    )
)

