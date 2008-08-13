import FWCore.ParameterSet.Config as cms

dtChamberEfficiencyMonitor = cms.EDFilter("DTChamberEfficiencyTask",
    debug = cms.untracked.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    minHitsSegment = cms.int32(5),
    minCloseDist = cms.double(20.0),
    minChi2NormSegment = cms.double(20.0)
)


