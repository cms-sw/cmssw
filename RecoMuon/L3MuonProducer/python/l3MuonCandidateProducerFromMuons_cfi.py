import FWCore.ParameterSet.Config as cms

l3MuonCandidateProducerFromMuons = cms.EDProducer("L3MuonCandidateProducerFromMuons",
    InputObjects = cms.InputTag("L2Muons")
)
