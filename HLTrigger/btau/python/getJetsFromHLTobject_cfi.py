import FWCore.ParameterSet.Config as cms

getJetsFromHLTobject = cms.EDProducer("GetJetsFromHLTobject",
    jets = cms.InputTag("hltBLifetimeL25filter")
)


