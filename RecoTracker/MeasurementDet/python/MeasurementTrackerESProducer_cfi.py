import FWCore.ParameterSet.Config as cms

MeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UseStripStripQualityDB = cms.bool(True), ## read Strip status from SiStripQuality

    OnDemand = cms.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(True), ## read APV and Fiber status from SiStripQuality
    MaskBadAPVFibers          = cms.bool(False),## if set to true, clusters entirely on bad APV and Fibers are ignored
                                                ## (UseStripAPVFiberQualityDB must also be true for this to work)
    DebugStripModuleQualityDB = cms.untracked.bool(False), ## dump out info om module status

    ComponentName = cms.string(''),
    stripClusterProducer = cms.string('siStripClusters'),
    Regional = cms.bool(False),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False), ## dump out info om module status

    #string StripCPE             = "SimpleStripCPE"
    # -- use simpleCPE untile the TkGluedMeasurementDet is 
    #    not corrected to handle properly the track direction
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False), ## dump out info om module status

    pixelClusterProducer = cms.string('siPixelClusters'),
    #stripLazyGetterProducer label only matters if Regional=true
    stripLazyGetterProducer = cms.string(''),
    # bool   UseStripCablingDB    = false     # NOT LONGER SUPPORTED, see below
    # bool   UseStripNoiseDB      = false     # NOT LONGER SUPPORTED, see below
    UseStripModuleQualityDB = cms.bool(True),
    PixelCPE = cms.string('PixelCPEfromTrackAngle')
)


