import FWCore.ParameterSet.Config as cms

hltESPTTRHBuilderPixelOnly = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('hltESPTTRHBuilderPixelOnly'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    Matcher = cms.string('StandardMatcher'),
    Phase2StripCPE = cms.string(''),
    PixelCPE = cms.string('PixelCPEGeneric'),
    StripCPE = cms.string('Fake')
)
# foo bar baz
# gqWuNy4Mqk8HP
# yXOOux7cvYrlD
