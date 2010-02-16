import FWCore.ParameterSet.Config as cms

profilerStart = cms.EDAnalyzer("Profiler",
    action = cms.int32(1),
    lastEvent = cms.int32(20),
    firstEvent = cms.int32(2)
)

profilerStop = cms.EDAnalyzer("Profiler",
    action = cms.int32(0),
    lastEvent = cms.int32(20),
    firstEvent = cms.int32(2)
)


