import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtOccupancyMonitor = DQMEDAnalyzer('DTEfficiencyTask',
    # switch for verbosity
    debug = cms.untracked.bool(False),
    # label for dtDigis
    digiLabel = cms.string('muonDTDigis'),
    # labels of 4D and 1D hits
    recHits4DLabel = cms.string('dt4DSegments'),
    recHitLabel = cms.string('dt1DRecHits'),
    # interval of lumi block after which we reset the histos
    ResetCycle = cms.untracked.int32(10000),
    # do all Layer vy Layer  plots or just summary ones
    doDetailedPlots = cms.untracked.bool(False)
)



