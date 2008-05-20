import FWCore.ParameterSet.Config as cms

L2MuonCandidates = cms.EDProducer("L2MuonCandidateProducer",
    InputObjects = cms.InputTag("L2Muons","UpdatedAtVtx")
)



