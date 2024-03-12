import FWCore.ParameterSet.Config as cms
globalMuonLinks = cms.EDProducer("MuonLinksProducer",
    inputCollection = cms.InputTag("muons","","@skipCurrentProcess")
)

# foo bar baz
# YFkUfDu5QnoUx
# AzN1tbKNKT5X5
