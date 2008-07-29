import FWCore.ParameterSet.Config as cms

hltMhtHtFilter = cms.EDFilter("HLTMhtHtFilter",
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    minMht = cms.double(100.0),
    minPtJet = cms.double(20.0)
)


