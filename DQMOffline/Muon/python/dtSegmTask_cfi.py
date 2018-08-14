import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtSegmentsMonitor = DQMEDAnalyzer('DTSegmentsTask',
    debug = cms.untracked.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    checkNoisyChannels = cms.untracked.bool(False)
)


