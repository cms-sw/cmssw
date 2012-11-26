import FWCore.ParameterSet.Config as cms

# instrument the menu with the modules and EndPath needed for timing studies
FastTimerService = cms.Service( 'FastTimerService',
    useRealTimeClock        = cms.untracked.bool( False ),
    enableTimingPaths       = cms.untracked.bool( True ),
    enableTimingModules     = cms.untracked.bool( False ),
    enableTimingSummary     = cms.untracked.bool( False ),
    skipFirstPath           = cms.untracked.bool( True ), 
    enableDQM               = cms.untracked.bool( True ),
    enableDQMbyPathActive   = cms.untracked.bool( False),
    enableDQMbyPathTotal    = cms.untracked.bool( True ),
    enableDQMbyPathOverhead = cms.untracked.bool( False ),
    enableDQMbyPathDetails  = cms.untracked.bool( True ),
    enableDQMbyPathCounters = cms.untracked.bool( True ),
    enableDQMbyModule       = cms.untracked.bool( False ),
    enableDQMbyLumi         = cms.untracked.bool( False ),
    dqmTimeRange            = cms.untracked.double(  10000 ),
    dqmTimeResolution       = cms.untracked.double(    10  ),
    dqmPathTimeRange        = cms.untracked.double(  10000 ),
    dqmPathTimeResolution   = cms.untracked.double(    10  ),
    dqmModuleTimeRange      = cms.untracked.double(   40.  ),
    dqmModuleTimeResolution = cms.untracked.double(    0.2 ),
    dqmPath                 = cms.untracked.string( "DQM/TimerService" )                                 
)
