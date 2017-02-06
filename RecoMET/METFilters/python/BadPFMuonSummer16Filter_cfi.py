import FWCore.ParameterSet.Config as cms

BadPFMuonSummer16Filter = cms.EDFilter(
    "BadPFMuonSummer16Filter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    muons  = cms.InputTag("muons"),   # Collection to test 
    taggingMode   = cms.bool(False),
    debug         = cms.bool(False),
    algo          = cms.int32(14),
    minDZ         = cms.double(0.1),              # dz threshold on PF muons to consider; this is not used
    minMuPt       = cms.double(100),               # pt threshold on PF muons 
    minTrkPtError  = cms.double(0.5),               # threshold on inner track pt Error
)
