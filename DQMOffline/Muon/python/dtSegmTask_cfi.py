import FWCore.ParameterSet.Config as cms

dtSegmentsMonitor = DQMStep1Module('DTSegmentsTask',
    debug = cms.untracked.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    checkNoisyChannels = cms.untracked.bool(False)
)


