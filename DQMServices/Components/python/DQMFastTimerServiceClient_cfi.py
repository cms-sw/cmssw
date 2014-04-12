import FWCore.ParameterSet.Config as cms

dqmFastTimerServiceClient = cms.EDAnalyzer('FastTimerServiceClient',
                                           dqmPath = cms.untracked.string( "DQM/TimerService" )
                                           )
