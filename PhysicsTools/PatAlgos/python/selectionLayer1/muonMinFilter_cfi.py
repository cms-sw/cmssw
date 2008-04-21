import FWCore.ParameterSet.Config as cms

minLayer1Muons = cms.EDFilter("PATMuonMinFilter",
    src = cms.InputTag("selectedLayer1Muons"),
    minNumber = cms.uint32(0)
)


