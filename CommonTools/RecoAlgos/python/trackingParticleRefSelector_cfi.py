import FWCore.ParameterSet.Config as cms

trackingParticleRefSelector = cms.EDFilter("TrackingParticleRefSelector",
    src = cms.InputTag("mix","MergedTrackTruth"),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    tip = cms.double(3.5),
    signalOnly = cms.bool(True),
    intimeOnly = cms.bool(False),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-2.4),
    lip = cms.double(30.0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.4),
    minHit = cms.int32(0)
)



