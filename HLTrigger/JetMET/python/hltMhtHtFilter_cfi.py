import FWCore.ParameterSet.Config as cms

hltMhtHtFilter = cms.EDFilter("HLTMhtHtFilter",
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTag = cms.untracked.bool( False ),
    minMht = cms.double(100.0),
    minPtJet = cms.double(20.0)
)


