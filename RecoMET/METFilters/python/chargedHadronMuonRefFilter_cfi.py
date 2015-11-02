import FWCore.ParameterSet.Config as cms

chargedHadronMuonRefFilter = cms.EDFilter(
    "ChargedHadronMuonRefFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    taggingMode   = cms.bool(False),
    ptMin         = cms.double(100.),               # Tracks with pT below this are ignored (will not be checked)
    verbose       = cms.untracked.bool(False),
    debug         = cms.bool(False),
)
