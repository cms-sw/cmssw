import FWCore.ParameterSet.Config as cms

MeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    ComponentName = cms.string(''),

    PixelCPE = cms.string('PixelCPEGeneric'),
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Phase2StripCPE = cms.string('Phase2StripCPEGeometric'),
    HitMatcher = cms.string('StandardMatcher'),

    SiStripQualityLabel         = cms.string(''),  ## unlabelled default SiStripQuality
    UseStripModuleQualityDB     = cms.bool(True),
    DebugStripModuleQualityDB   = cms.untracked.bool(False), ## dump out info om module status
    UseStripAPVFiberQualityDB   = cms.bool(True),            ## read APV and Fiber status from SiStripQuality
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False), ## dump out info om module status
    MaskBadAPVFibers            = cms.bool(True),            ## if set to true, clusters with barycenter on bad APV and Fibers are ignored
                                                             ## (UseStripAPVFiberQualityDB must also be true for this to work)
    UseStripStripQualityDB      = cms.bool(True),            ## read Strip status from SiStripQuality
    DebugStripStripQualityDB    = cms.untracked.bool(False), ## dump out info om module status
    badStripCuts  = cms.PSet(
        TIB = cms.PSet( maxBad = cms.uint32(4), maxConsecutiveBad = cms.uint32(2) ),
        TID = cms.PSet( maxBad = cms.uint32(4), maxConsecutiveBad = cms.uint32(2) ),
        TOB = cms.PSet( maxBad = cms.uint32(4), maxConsecutiveBad = cms.uint32(2) ),
        TEC = cms.PSet( maxBad = cms.uint32(4), maxConsecutiveBad = cms.uint32(2) ),
    ),

    UsePixelModuleQualityDB   = cms.bool(True),            ## Use DB info at the module level (that is, detid level)
    DebugPixelModuleQualityDB = cms.untracked.bool(False), ## dump out info om module status
    UsePixelROCQualityDB      = cms.bool(True),            ## Use DB info at the ROC level
    DebugPixelROCQualityDB    = cms.untracked.bool(False), ## dump out info om module status
)


