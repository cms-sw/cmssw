import FWCore.ParameterSet.Config as cms

OnDemandMeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    #string StripCPE             = "StripCPEfromTrackAngle"
    StripCPE = cms.string('SimpleStripCPE'),
    #stripLazyGetterProducer label only matters if Regional=true
    stripLazyGetterProducer = cms.string('SiStripRawToClustersFacility'),
    UseStripNoiseDB = cms.bool(False),
    OnDemand = cms.bool(True),
    ComponentName = cms.string('OnDemandMeasurementTracker'),
    stripClusterProducer = cms.string('measurementTrackerSiStripRefGetterProducer'),
    Regional = cms.bool(True),
    UseStripCablingDB = cms.bool(False),
    pixelClusterProducer = cms.string('siPixelClusters'),
    # -- use simpleCPE untile the TkGluedMeasurementDet is 
    #    not corrected to handle properly the track direction
    HitMatcher = cms.string('StandardMatcher'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle')
)


