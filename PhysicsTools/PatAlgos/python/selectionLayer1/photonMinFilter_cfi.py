import FWCore.ParameterSet.Config as cms

# module to filter on the minimal number of Photons
minLayer1Photons = cms.EDFilter("PATPhotonMinFilter",
    src = cms.InputTag("selectedLayer1Photons"),
    minNumber = cms.uint32(0)
)


