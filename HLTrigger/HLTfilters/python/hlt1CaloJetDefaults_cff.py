import FWCore.ParameterSet.Config as cms

hlt1CaloJetDefaults = cms.PSet(
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("iterativeCone5CaloJets"),
    MinPt = cms.double(100.0),
    MinN = cms.int32(1)
)

