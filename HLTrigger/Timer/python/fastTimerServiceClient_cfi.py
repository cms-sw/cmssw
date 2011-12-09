import FWCore.ParameterSet.Config as cms

# FastTimerServiceClient
fastTimerServiceClient = cms.EDAnalyzer( 'FastTimerServiceClient',
    dqmPath = cms.untracked.string( "HLT/TimerService" )
)
