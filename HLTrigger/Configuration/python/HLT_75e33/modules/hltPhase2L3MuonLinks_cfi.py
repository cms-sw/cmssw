import FWCore.ParameterSet.Config as cms

#Extract global-muon links 
hltPhase2L3MuonLinks = cms.EDProducer("MuonLinksProducer",
    inputCollection = cms.InputTag("hltPhase2L3Muons")
)
