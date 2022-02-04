import FWCore.ParameterSet.Config as cms

hltL2MuonFromL1TkMuonCandidates = cms.EDProducer("L2MuonCandidateProducer",
    InputObjects = cms.InputTag("hltL2MuonsFromL1TkMuon","UpdatedAtVtx")
)
