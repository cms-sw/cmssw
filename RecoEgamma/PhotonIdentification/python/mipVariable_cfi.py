import FWCore.ParameterSet.Config as cms


mipVariable = cms.PSet(
    #required inputs
    ComponentName = cms.string('mipVariable'),

    barrelEcalRecHitCollection = cms.InputTag('ecalRecHit:EcalRecHitsEB'),
    endcapEcalRecHitCollection = cms.InputTag('ecalRecHit:EcalRecHitsEE'),

    #some of the variables use to get mipHalo bool
    # These values could be improved with further study
    YRangeFit          = cms.double(7.0),
    XRangeFit          = cms.double(180.0),
    ResidualWidth      = cms.double(0.23),
    HaloDiscThreshold  = cms.double(70.0)
)


