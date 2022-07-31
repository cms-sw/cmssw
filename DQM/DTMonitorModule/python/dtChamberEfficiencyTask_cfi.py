import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtChamberEfficiencyMonitor = DQMEDAnalyzer('DTChamberEfficiencyTask',
    # define the segment quality
    minHitsSegment = cms.int32(5),
    # parameter for check on extrapolated check
    minCloseDist = cms.double(20.0),
    # labels of 4D segments
    recHits4DLabel = cms.untracked.string('dt4DSegments'),
    # switch for verbosity
    debug = cms.untracked.bool(False),
    minChi2NormSegment = cms.double(20.0),
    # interval of lumi block after which we reset the histos
    ResetCycle = cms.untracked.int32(10000),
    # the running mode
    onlineMonitor = cms.untracked.bool(False),
    # the analysis mode
    detailedAnalysis = cms.untracked.bool(False)                                       
)


