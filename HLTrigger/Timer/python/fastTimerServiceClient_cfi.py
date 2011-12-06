import FWCore.ParameterSet.Config as cms

# FastTimerServiceClient
fastTimerServiceClient = cms.EDAnalyzer( 'FastTimerServiceClient',
    hltProcessName      = cms.untracked.string( "@" ),
    dqmPath             = cms.untracked.string( "HLT/TimerService" ),
)
