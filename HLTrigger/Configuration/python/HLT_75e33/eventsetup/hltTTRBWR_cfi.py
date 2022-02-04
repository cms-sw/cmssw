import FWCore.ParameterSet.Config as cms

hltTTRBWR = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('hltESPTTRHBWithTrackAngle'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    Matcher = cms.string('StandardMatcher'),
    Phase2StripCPE = cms.string('Phase2StripCPE'),
    PixelCPE = cms.string('PixelCPEGeneric'),
    StripCPE = cms.string('FakeStripCPE')
)
