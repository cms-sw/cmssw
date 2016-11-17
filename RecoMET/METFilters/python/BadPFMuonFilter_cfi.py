import FWCore.ParameterSet.Config as cms

BadPFMuonFilter = cms.EDFilter(
    "BadPFMuonFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    muons  = cms.InputTag("muons"),   # Collection to test 
    taggingMode   = cms.bool(False),
    debug         = cms.bool(False),
    algo          = cms.int32(14),
    minDZ         = cms.double(0.1),              # dz threshold on PF muons to consider; this is not used
    minMuPt       = cms.double(100),               # pt threshold on PF muons 
    minPtError    = cms.double(2.0),               # threshold on best track RelptError
    innerTrackRelErr = cms.double(1.0),            # threshold on innerTrack relPtErr
    segmentCompatibility = cms.double(0.3),        # compatibility between the inner track and the segments in the muon spectrometer
)
