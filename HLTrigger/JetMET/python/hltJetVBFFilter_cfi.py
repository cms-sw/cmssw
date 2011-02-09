import FWCore.ParameterSet.Config as cms

hltJetVBFFilter = cms.EDFilter("HLTJetVBFFilter",
    minDeltaEta = cms.double(2.0),
    minEtLow = cms.double(20.0),
    minEtHigh = cms.double(20.0),                               
    saveTag = cms.untracked.bool( False ),
    inputTag = cms.InputTag("iterativeCone5CaloJets")
)


