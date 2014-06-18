import FWCore.ParameterSet.Config as cms

# FastTimerService client
import HLTrigger.Timer.fastTimerServiceClient_cfi as __fastTimerServiceClient_cfi
dqmFastTimerServiceClient = __fastTimerServiceClient_cfi.fastTimerServiceClient.clone()
dqmFastTimerServiceClient.dqmPath = cms.untracked.string( "DQM/TimerService" )
