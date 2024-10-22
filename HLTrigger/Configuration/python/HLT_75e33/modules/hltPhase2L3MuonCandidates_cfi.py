import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonCandidates = cms.EDProducer("L3MuonCandidateProducerFromMuons",
    InputObjects = cms.InputTag("hltPhase2L3Muons")
)
