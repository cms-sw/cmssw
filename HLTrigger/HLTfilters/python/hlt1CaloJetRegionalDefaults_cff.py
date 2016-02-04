import FWCore.ParameterSet.Config as cms

hlt1CaloJetRegionalDefaults = cms.PSet(
    saveTags = cms.bool( False ),
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("iterativeCone5CaloJetsRegional"),
    MinPt = cms.double(100.0),
    MinN = cms.int32(1)
)

