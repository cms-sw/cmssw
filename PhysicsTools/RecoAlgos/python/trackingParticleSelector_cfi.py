import FWCore.ParameterSet.Config as cms

trackingParticleSelector = cms.EDFilter("TrackingParticleSelector",
    src = cms.InputTag("mix","MergedTrackTruth"),
    chargedOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    pdgId = cms.vint32(),
    tip = cms.double(3.5),
    signalOnly = cms.bool(True),
    minRapidity = cms.double(-2.4),
    lip = cms.double(30.0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.4),
    minHit = cms.int32(0)
)



