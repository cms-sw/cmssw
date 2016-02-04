import FWCore.ParameterSet.Config as cms

allMuons = cms.EDProducer("MuonShallowCloneProducer",
    src = cms.InputTag("muons")
)


