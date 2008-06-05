import FWCore.ParameterSet.Config as cms

dtChamberEfficiencyMonitor = cms.EDFilter("DTChamberEfficiencyTask",
    # define the segment quality
    minHitsSegment = cms.int32(5),
    # parameter for check on extrapolated check
    minCloseDist = cms.double(20.0),
    # labels of 4D segments
    recHits4DLabel = cms.string('dt4DSegments'),
    # switch for verbosity
    debug = cms.untracked.bool(False),
    minChi2NormSegment = cms.double(20.0),
    # interval of lumi block after which we reset the histos
    ResetCycle = cms.untracked.int32(10000)
)


