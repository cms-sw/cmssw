import FWCore.ParameterSet.Config as cms

l1t4PFPuppiJet25OnlineMaxEta2p4 = cms.EDFilter("L1TJetFilter",
    MaxEta = cms.double(2.4),
    MinEta = cms.double(-2.4),
    MinN = cms.int32(4),
    MinPt = cms.double(25.0),
    inputTag = cms.InputTag("l1tSlwPFPuppiJetsCorrected","Phase1L1TJetFromPfCandidates")
)
