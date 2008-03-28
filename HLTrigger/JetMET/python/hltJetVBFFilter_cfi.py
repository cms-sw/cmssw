import FWCore.ParameterSet.Config as cms

hltJetVBFFilter = cms.EDFilter("HLTJetVBFFilter",
    minDeltaEta = cms.double(4.2),
    minEt = cms.double(40.0),
    inputTag = cms.InputTag("iterativeCone5CaloJets")
)


