import FWCore.ParameterSet.Config as cms

# FastTimerService
FastTimerService = cms.Service( 'FastTimerService',
    useRealTimeClock    = cms.untracked.bool( False ),
    enableTimingModules = cms.untracked.bool( True ),
    enableTimingPaths   = cms.untracked.bool( True ),
    enableTimingSummary = cms.untracked.bool( True ),
    enableDQM           = cms.untracked.bool( True ),
    enableDQMbyLumi     = cms.untracked.bool( False ),
    skipFirstPath       = cms.untracked.bool( False ), 
    dqmEventTimeRange        = cms.untracked.double( 1000. ),
    dqmEventTimeResolution   = cms.untracked.double(   5. ),
    dqmPathTimeRange        = cms.untracked.double( 100. ),
    dqmPathTimeResolution   = cms.untracked.double(   0.5 ),
    dqmModuleTimeRange        = cms.untracked.double( 40. ),
    dqmModuleTimeResolution   = cms.untracked.double(   0.2 ),
    dqmPath             = cms.untracked.string( "HLT/TimerService" )
)
