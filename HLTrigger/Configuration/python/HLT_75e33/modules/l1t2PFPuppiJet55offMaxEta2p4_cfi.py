import FWCore.ParameterSet.Config as cms

l1t2PFPuppiJet55offMaxEta2p4 = cms.EDFilter("L1TJetFilter",
    MaxEta = cms.double(2.4),
    MinEta = cms.double(-2.4),
    MinN = cms.int32(2),
    MinPt = cms.double(55.0),
    Scalings = cms.PSet(
        barrel = cms.vdouble(11.1254, 1.40627, 0),
        endcap = cms.vdouble(42.4039, 1.33052, 0),
        overlap = cms.vdouble(24.8375, 1.4152, 0)
    ),
    inputTag = cms.InputTag("l1tSlwPFPuppiJetsCorrected","Phase1L1TJetFromPfCandidates")
)
