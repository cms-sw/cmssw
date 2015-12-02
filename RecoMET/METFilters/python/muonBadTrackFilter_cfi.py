import FWCore.ParameterSet.Config as cms

muonBadTrackFilter = cms.EDFilter(
    "MuonBadTrackFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    taggingMode   = cms.bool(False),
    ptMin         = cms.double(100.),               # Tracks with pT below this are ignored (will not be checked)
    chi2Min       = cms.double(100.),               # global Tracks with normalizeChi2 below this are ignored (will not be checked)
    p1            = cms.double(5.),                 # parameter 1 of calo resolution formula
    p2            = cms.double(1.2),                # parameter 2 of calo resolution formula
    p3            = cms.double(0.06),               # parameter 3 of calo resolution formula
    verbose       = cms.untracked.bool(False),
    debug         = cms.bool(False),
)
