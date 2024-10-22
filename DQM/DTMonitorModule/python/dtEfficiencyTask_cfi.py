import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtEfficiencyMonitor = DQMEDAnalyzer('DTEfficiencyTask',
    # switch for verbosity
    debug = cms.untracked.bool(False),
    # labels of 4D and 1D hits
    recHits4DLabel = cms.untracked.string('dt4DSegments'),
    recHitLabel = cms.untracked.string('dt1DRecHits'),
    # interval of lumi block after which we reset the histos
    ResetCycle = cms.untracked.int32(10000)
)


