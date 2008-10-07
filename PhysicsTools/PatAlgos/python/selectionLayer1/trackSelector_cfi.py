import FWCore.ParameterSet.Config as cms

selectedLayer1TrackCands = cms.EDFilter("PATGenericParticleSelector",
    src = cms.InputTag("allLayer1TrackCands"),
    cut = cms.string('pt > 0. & abs(eta) < 12.')
)


