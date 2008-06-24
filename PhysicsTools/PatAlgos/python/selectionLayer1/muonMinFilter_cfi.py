import FWCore.ParameterSet.Config as cms

# module to filter on the minimal number of Muons
minLayer1Muons = cms.EDFilter("PATMuonMinFilter",
    src = cms.InputTag("selectedLayer1Muons"),
    minNumber = cms.uint32(0)
)


