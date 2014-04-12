import FWCore.ParameterSet.Config as cms
globalMuonLinks = cms.EDProducer("MuonLinksProducer",
    inputCollection = cms.InputTag("muons")
)

