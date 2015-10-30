import FWCore.ParameterSet.Config as cms

chargedHadronMuonRefFilter = cms.EDFilter(
    "ChargedHadronMuonRefFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    ptMin         = cms.double(10.),               # Tracks with pT below this are ignored (will not be checked)
    verbose       = cms.untracked.bool(True),
    debug         = cms.bool(True),
)
