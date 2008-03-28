import FWCore.ParameterSet.Config as cms

profilerStart = cms.EDFilter("Profiler",
    action = cms.int32(1),
    lastEvent = cms.int32(20),
    firstEvent = cms.int32(2)
)

profilerStop = cms.EDFilter("Profiler",
    action = cms.int32(0),
    lastEvent = cms.int32(20),
    firstEvent = cms.int32(2)
)


