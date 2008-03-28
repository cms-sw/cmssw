import FWCore.ParameterSet.Config as cms

zToMuMuRECOEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCandidatesOwned_allMuons_*_*', 'keep recoCandidatesOwned_allTracks_*_*', 'keep recoCandidatesOwned_allStandAloneMuonTracks_*_*', 'keep *_allMuonIsolations_*_*', 'keep *_allTrackIsolations_*_*', 'keep *_allStandAloneMuonTrackIsolations_*_*', 'keep *_zToMuMu_*_*', 'keep *_zToMuMuOneTrack_*_*', 'keep *_zToMuMuOneStandAloneMuonTrack_*_*', 'keep *_allMuonsGenParticlesMatch_*_*', 'keep *_allTracksGenParticlesLeptonMatch_*_*', 'keep *_allStandAloneMuonTracksGenParticlesLeptonMatch_*_*', 'keep *_zToMuMuGenParticlesMatch_*_*', 'keep *_zToMuMuOneTrackGenParticlesMatch_*_*', 'keep *_zToMuMuOneStandAloneMuonTrackGenParticlesMatch_*_*')
)
zToMuMuRECOEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToMuMuRECOPath', 'zToMuMuOneTrackRECOPath', 'zToMuMuOneStandAloneMuonTrackRECOPath')
    )
)

