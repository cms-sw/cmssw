import FWCore.ParameterSet.Config as cms

MeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),

    OnDemand = cms.bool(False),

    ComponentName = cms.string(''),
    stripClusterProducer = cms.string('siStripClusters'),
    Regional = cms.bool(False),

    # -- use simpleCPE untile the TkGluedMeasurementDet is 
    #    not corrected to handle properly the track direction
    HitMatcher = cms.string('StandardMatcher'),

    UseStripModuleQualityDB     = cms.bool(True),
    DebugStripModuleQualityDB   = cms.untracked.bool(False), ## dump out info om module status
    UseStripAPVFiberQualityDB   = cms.bool(True),            ## read APV and Fiber status from SiStripQuality
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False), ## dump out info om module status
    MaskBadAPVFibers            = cms.bool(False),           ## if set to true, clusters entirely on bad APV and Fibers are ignored
                                                             ## (UseStripAPVFiberQualityDB must also be true for this to work)
    UseStripStripQualityDB      = cms.bool(True),            ## read Strip status from SiStripQuality
    DebugStripStripQualityDB    = cms.untracked.bool(False), ## dump out info om module status

    UsePixelModuleQualityDB   = cms.bool(True),            ## Use DB info at the module level (that is, detid level)
    DebugPixelModuleQualityDB = cms.untracked.bool(False), ## dump out info om module status
    UsePixelROCQualityDB      = cms.bool(True),            ## Use DB info at the ROC level
    DebugPixelROCQualityDB    = cms.untracked.bool(False), ## dump out info om module status

    pixelClusterProducer = cms.string('siPixelClusters'),
    #stripLazyGetterProducer label only matters if Regional=true
    stripLazyGetterProducer = cms.string(''),
    PixelCPE = cms.string('PixelCPEGeneric')
)


