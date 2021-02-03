import FWCore.ParameterSet.Config as cms

ttrhbwor = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('WithoutRefit'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    Matcher = cms.string('Fake'),
    Phase2StripCPE = cms.string('Phase2StripCPE'),
    PixelCPE = cms.string('Fake'),
    StripCPE = cms.string('Fake')
)
