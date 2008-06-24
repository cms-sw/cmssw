import FWCore.ParameterSet.Config as cms

# module to filter on the maximal number of Muons
maxLayer1Muons = cms.EDFilter("PATMuonMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Muons")
)


