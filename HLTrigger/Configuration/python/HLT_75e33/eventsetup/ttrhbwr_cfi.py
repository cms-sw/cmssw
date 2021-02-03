import FWCore.ParameterSet.Config as cms

ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('WithTrackAngle'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    Matcher = cms.string('StandardMatcher'),
    Phase2StripCPE = cms.string('Phase2StripCPE'),
    PixelCPE = cms.string('PixelCPEGeneric'),
    StripCPE = cms.string('StripCPEfromTrackAngle')
)
