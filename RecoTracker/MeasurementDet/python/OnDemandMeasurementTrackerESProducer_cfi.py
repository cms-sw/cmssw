import FWCore.ParameterSet.Config as cms

OnDemandMeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    skipClusters = cms.InputTag(""),
    ComponentName = cms.string('OnDemandMeasurementTracker'),

    OnDemand = cms.bool(True),
    Regional = cms.bool(True),

    pixelClusterProducer = cms.string('siPixelClusters'),
    stripClusterProducer = cms.string('measurementTrackerSiStripRefGetterProducer'),
    #stripLazyGetterProducer label only matters if Regional=true
    stripLazyGetterProducer = cms.string('SiStripRawToClustersFacility'),

    PixelCPE = cms.string('PixelCPEGeneric'),
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    HitMatcher = cms.string('StandardMatcher'),

    UseStripCablingDB = cms.bool(False),
    UseStripNoiseDB = cms.bool(False),

    inactivePixelDetectorLabels = cms.VInputTag(),
    inactiveStripDetectorLabels = cms.VInputTag(),
)


