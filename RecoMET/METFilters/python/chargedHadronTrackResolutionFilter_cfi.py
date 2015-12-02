import FWCore.ParameterSet.Config as cms

chargedHadronTrackResolutionFilter = cms.EDFilter(
    "ChargedHadronTrackResolutionFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    PFMET         = cms.InputTag("pfMet"),   # Collection to test
    taggingMode   = cms.bool(False),
    ptMin         = cms.double(100.),               # Tracks with pT below this are ignored (will not be checked)
    MetSignifMin  = cms.double(5.),                 # minimum relative MET significance change when removing charged hadron from MET
    verbose       = cms.untracked.bool(False),
    debug         = cms.bool(False),
)
