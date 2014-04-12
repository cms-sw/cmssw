import FWCore.ParameterSet.Config as cms

dtSegmentsMonitor = cms.EDAnalyzer("DTSegmentsTask",
    debug = cms.untracked.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    checkNoisyChannels = cms.untracked.bool(False),
    OutputFileName = cms.string('SegmentsMonitoring.root'),
    OutputMEsInRootFile = cms.bool(False)
)


