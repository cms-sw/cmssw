import FWCore.ParameterSet.Config as cms

L3MuonCandidates = cms.EDProducer("L3MuonCandidateProducer",
    InputObjects = cms.InputTag("L3Muons"),
    InputLinksObjects = cms.InputTag(""),
    MuonPtOption = cms.string("Global")
)
