import FWCore.ParameterSet.Config as cms

hltJetVBFFilter = cms.EDFilter("HLTJetVBFFilter",
    inputTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTags = cms.bool( False ),
    minEtLow = cms.double(20.0),
    minEtHigh = cms.double(20.0),                               
    etaOpposite = cms.bool(False),
    minDeltaEta = cms.double(2.0),
    minInvMass = cms.double(0.0),
    maxEta = cms.double(-1.0)                               
)
