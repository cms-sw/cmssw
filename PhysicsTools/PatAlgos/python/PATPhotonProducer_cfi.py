import FWCore.ParameterSet.Config as cms

allLayer1Photons = cms.EDProducer("PATPhotonProducer",
    # General configurables
    photonSource = cms.InputTag("allLayer0Photons")
)


