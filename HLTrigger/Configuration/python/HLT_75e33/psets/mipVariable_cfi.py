import FWCore.ParameterSet.Config as cms

mipVariable = cms.PSet(
    ComponentName = cms.string('mipVariable'),
    HaloDiscThreshold = cms.double(70.0),
    ResidualWidth = cms.double(0.23),
    XRangeFit = cms.double(180.0),
    YRangeFit = cms.double(7.0),
    barrelEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)