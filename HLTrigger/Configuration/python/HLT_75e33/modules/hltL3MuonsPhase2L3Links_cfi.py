import FWCore.ParameterSet.Config as cms

hltL3MuonsPhase2L3Links = cms.EDProducer("MuonLinksProducer",
    inputCollection = cms.InputTag("hltPhase2L3Muons")
)
