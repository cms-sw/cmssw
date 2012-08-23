import FWCore.ParameterSet.Config as cms

# FastTimerService
FastTimerService = cms.Service( 'FastTimerService',
    useRealTimeClock        = cms.untracked.bool( True ),
    enableTimingPaths       = cms.untracked.bool( True ),
    enableTimingModules     = cms.untracked.bool( True ),
    enableTimingSummary     = cms.untracked.bool( False ),
    skipFirstPath           = cms.untracked.bool( False ), 
    enableDQM               = cms.untracked.bool( True ),
    enableDQMbyPathActive   = cms.untracked.bool( False ),
    enableDQMbyPathTotal    = cms.untracked.bool( True ),
    enableDQMbyPathOverhead = cms.untracked.bool( False ),
    enableDQMbyPathDetails  = cms.untracked.bool( False ),
    enableDQMbyPathCounters = cms.untracked.bool( True ),
    enableDQMbyModule       = cms.untracked.bool( False ),
    enableDQMbyLumi         = cms.untracked.bool( False ),
    dqmTimeRange            = cms.untracked.double( 1000.  ),
    dqmTimeResolution       = cms.untracked.double(    5.  ),
    dqmPathTimeRange        = cms.untracked.double(  100.  ),
    dqmPathTimeResolution   = cms.untracked.double(    0.5 ),
    dqmModuleTimeRange      = cms.untracked.double(   40.  ),
    dqmModuleTimeResolution = cms.untracked.double(    0.2 ),
    dqmPath                 = cms.untracked.string( "HLT/TimerService" )
)
