import FWCore.ParameterSet.Config as cms

OnDemandMeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    ComponentName = cms.string('OnDemandMeasurementTracker'),

    OnDemand = cms.bool(True),
    Regional = cms.bool(True),

    PixelCPE = cms.string('PixelCPEGeneric'),
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    HitMatcher = cms.string('StandardMatcher'),

    UseStripCablingDB = cms.bool(False),
    UseStripNoiseDB = cms.bool(False),
)


