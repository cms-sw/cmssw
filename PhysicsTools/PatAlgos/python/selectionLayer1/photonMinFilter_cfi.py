import FWCore.ParameterSet.Config as cms

minLayer1Photons = cms.EDFilter("PATPhotonMinFilter",
    src = cms.InputTag("selectedLayer1Photons"),
    minNumber = cms.uint32(0)
)


