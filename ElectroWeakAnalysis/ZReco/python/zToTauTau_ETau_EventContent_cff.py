import FWCore.ParameterSet.Config as cms

# S.G: STILL NEED TO BE COMPLETED
zToTauTauETauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoCandidatesOwned_allElectrons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allSuperClusters_*_*', 
        'keep *_allElectronIsolations_*_*', 
        'keep *_allTrackIsolations_*_*', 
        'keep *_allSuperClusterIsolations_*_*', 
        'keep *_allElectronsGenParticlesMatch_*_*', 
        'keep *_allTracksGenParticlesLeptonMatch_*_*', 
        'keep *_allSuperClustersGenParticlesLeptonMatch_*_*')
)
zToTauTauETauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToTauTauETauHLTPath')
    )
)

