# The following comments couldn't be translated into the new config version:

#    "keep *_goodMuonIsolations_*_*",
#    "keep *_goodTrackIsolations_*_*",
#    "keep *_goodStandAloneMuonTrackIsolations_*_*",

#    "keep *_goodMuonMCMatch_*_*",
#    "keep *_goodTrackMCMatch_*_*",
#    "keep *_goodStandAloneMuonTrackMCMatch_*_*",
#    "keep *_goodZToMuMuMCMatch_*_*",
#    "keep *_goodZToMuMuOneTrackMCMatch_*_*",
#    "keep *_goodZToMuMuOneStandAloneMuonTrackMCMatch_*_*",

import FWCore.ParameterSet.Config as cms

zToMuMuNewEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticles_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep *_goodMuons_*_*', 'keep *_goodTracks_*_*', 'keep *_goodStandAloneMuonTracks_*_*', 'keep *_muonIsolations_*_*', 'keep *_goodZToMuMu_*_*', 'keep *_goodZToMuMuOneTrack_*_*', 'keep *_goodZToMuMuOneStandAloneMuonTrack_*_*', 'keep *_goodZMCMatch_*_*', 'drop *_*_*_HLT')
)
zToMuMuNewEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToMuMuNewPath', 'zToMuMuOneTrackNewPath', 'zToMuMuOneStandAloneMuonTrackNewPath')
    )
)

