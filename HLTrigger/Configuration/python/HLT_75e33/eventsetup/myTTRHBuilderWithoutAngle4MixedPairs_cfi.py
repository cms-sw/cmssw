import FWCore.ParameterSet.Config as cms

myTTRHBuilderWithoutAngle4MixedPairs = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    Matcher = cms.string('StandardMatcher'),
    Phase2StripCPE = cms.string('Phase2StripCPE'),
    PixelCPE = cms.string('PixelCPEGeneric'),
    StripCPE = cms.string('Fake')
)
