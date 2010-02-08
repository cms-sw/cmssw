import FWCore.ParameterSet.Config as cms

hltJetVBFFilter = cms.EDFilter("HLTJetVBFFilter",
    minDeltaEta = cms.double(4.2),
    minEt = cms.double(40.0),
    saveTag = cms.untracked.bool( False ),
    inputTag = cms.InputTag("iterativeCone5CaloJets")
)


