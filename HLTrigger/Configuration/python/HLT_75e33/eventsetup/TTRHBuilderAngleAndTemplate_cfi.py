import FWCore.ParameterSet.Config as cms

TTRHBuilderAngleAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('WithAngleAndTemplate'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    Matcher = cms.string('StandardMatcher'),
    Phase2StripCPE = cms.string('Phase2StripCPE'),
    PixelCPE = cms.string('PixelCPETemplateReco'),
    StripCPE = cms.string('StripCPEfromTrackAngle')
)
