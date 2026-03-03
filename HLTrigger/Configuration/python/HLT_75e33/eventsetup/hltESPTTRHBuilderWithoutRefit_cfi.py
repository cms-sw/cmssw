import FWCore.ParameterSet.Config as cms

hltESPTTRHBuilderWithoutRefit = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('hltESPTTRHBuilderWithoutRefit'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    Matcher = cms.string('Fake'),
    Phase2StripCPE = cms.string(''),
    PixelCPE = cms.string('Fake'),
    StripCPE = cms.string('Fake')
)
