import FWCore.ParameterSet.Config as cms

maxLayer1Muons = cms.EDFilter("PATMuonMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Muons")
)


