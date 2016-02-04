import FWCore.ParameterSet.Config as cms

allMuons = cms.EDProducer("MuonCloneProducer",
    src = cms.InputTag("muons")
)


