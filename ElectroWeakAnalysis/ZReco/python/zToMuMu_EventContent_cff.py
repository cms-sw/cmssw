import FWCore.ParameterSet.Config as cms

zToMuMuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticles_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep *_goodMuons_*_*', 'keep *_goodTracks_*_*', 'keep *_goodStandAloneMuonTracks_*_*', 'keep *_muonIsolations_*_*', 'keep *_goodZToMuMu_*_*', 'keep *_goodZToMuMuOneTrack_*_*', 'keep *_goodZToMuMuOneStandAloneMuonTrack_*_*', 'keep *_goodZMCMatch_*_*', 'drop *_*_*_HLT')
)
zToMuMuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToMuMuPath', 'zToMuMuOneTrackPath', 'zToMuMuOneStandAloneMuonTrackPath')
    )
)

