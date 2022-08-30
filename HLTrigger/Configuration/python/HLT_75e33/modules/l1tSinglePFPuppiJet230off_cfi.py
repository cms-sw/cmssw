import FWCore.ParameterSet.Config as cms

l1tSinglePFPuppiJet230off = cms.EDFilter("L1TJetFilter",
    MaxEta = cms.double(5.0),
    MinEta = cms.double(-5.0),
    MinN = cms.int32(1),
    MinPt = cms.double(230.0),
    Scalings = cms.PSet(
        barrel = cms.vdouble(11.1254, 1.40627, 0),
        endcap = cms.vdouble(42.4039, 1.33052, 0),
        overlap = cms.vdouble(24.8375, 1.4152, 0)
    ),
    inputTag = cms.InputTag("Phase1L1TJetCalibrator9x9","Phase1L1TJetFromPfCandidates")
)
