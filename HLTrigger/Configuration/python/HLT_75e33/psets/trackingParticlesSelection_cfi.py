import FWCore.ParameterSet.Config as cms

trackingParticlesSelection = cms.PSet(
    chargedOnlyTP = cms.bool(True),
    intimeOnlyTP = cms.bool(False),
    lipTP = cms.double(1000),
    maxRapidityTP = cms.double(5.0),
    minHitTP = cms.int32(0),
    minRapidityTP = cms.double(-5.0),
    pdgIdTP = cms.vint32(),
    ptMaxTP = cms.double(1e+100),
    ptMinTP = cms.double(0.1),
    signalOnlyTP = cms.bool(False),
    stableOnlyTP = cms.bool(False),
    tipTP = cms.double(1000)
)