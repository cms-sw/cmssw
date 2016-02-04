import FWCore.ParameterSet.Config as cms

hltForwardBackwardJetsFilter = cms.EDFilter("HLTForwardBackwardJetsFilter",
    inputTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTag = cms.untracked.bool( False ),
    minPt = cms.double(15.0),
    minEta = cms.double(3.),
    maxEta = cms.double(5.1)
)


