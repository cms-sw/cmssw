import FWCore.ParameterSet.Config as cms

selectedLayer1Photons = cms.EDFilter("PATPhotonSelector",
    src = cms.InputTag("allLayer1Photons"),
    cut = cms.string('pt > 10. & abs(eta) < 2.4')
)


