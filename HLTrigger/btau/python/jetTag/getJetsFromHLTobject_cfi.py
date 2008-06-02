import FWCore.ParameterSet.Config as cms

getJetsFromHLTobject = cms.EDFilter("GetJetsFromHLTobject",
    jets = cms.InputTag("hltBLifetimeL25filter")
)


