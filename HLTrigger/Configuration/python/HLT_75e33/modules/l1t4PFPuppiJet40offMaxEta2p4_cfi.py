import FWCore.ParameterSet.Config as cms

l1t4PFPuppiJet40offMaxEta2p4 = cms.EDFilter("L1TJetFilter",
    MaxEta = cms.double(2.4),
    MinEta = cms.double(-2.4),
    MinN = cms.int32(4),
    MinPt = cms.double(40.0),
    Scalings = cms.PSet(
        barrel = cms.vdouble(11.1254, 1.40627, 0),
        endcap = cms.vdouble(42.4039, 1.33052, 0),
        overlap = cms.vdouble(24.8375, 1.4152, 0)
    ),
    inputTag = cms.InputTag("l1tPhase1JetCalibrator9x9trimmed","Phase1L1TJetFromPfCandidates")
)
