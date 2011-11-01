import FWCore.ParameterSet.Config as cms

# FastTimerService
FastTimerService = cms.Service( 'FastTimerService',
  useRealTimeClock    = cms.untracked.bool( False ),
  enableTimingModules = cms.untracked.bool( False ),
  enableTimingPaths   = cms.untracked.bool( True ),
  enableTimingSummary = cms.untracked.bool( True ),
  enableDQM           = cms.untracked.bool( True ),
  enableDQMbyLumi     = cms.untracked.bool( False ),
  dqmTimeRange        = cms.untracked.double( 1000. ),
  dqmTimeResolution   = cms.untracked.double(    5. ),
  dqmPath             = cms.untracked.string( "HLT/TimerService" ),
)
