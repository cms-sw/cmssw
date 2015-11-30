import FWCore.ParameterSet.Config as cms

muonBadTrackFilter = cms.EDFilter(
    "MuonBadTrackFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    taggingMode   = cms.bool(False),
    ptMin         = cms.double(100.),               # Tracks with pT below this are ignored (will not be checked)
    chi2Min       = cms.double(100.),               # global Tracks with normalizeChi2 below this are ignored (will not be checked)
    verbose       = cms.untracked.bool(False),
    debug         = cms.bool(False),
)
