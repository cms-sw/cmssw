import FWCore.ParameterSet.Config as cms

BadPFMuonFilter = cms.EDFilter(
    "BadPFMuonFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    taggingMode   = cms.bool(False),
    debug         = cms.bool(False),
    minDZ         = cms.double(0.1),              # dz threshold on PF muons to consider
    minMuPt       = cms.double(100),               # pt threshold on PF muons 
    minTrkPtError  = cms.double(-1),               # threshold on inner track pt Error 
)
