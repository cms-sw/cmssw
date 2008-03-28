import FWCore.ParameterSet.Config as cms

# module to filter on the maximal number of Photons
maxLayer1Photons = cms.EDFilter("PATPhotonMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Photons")
)


