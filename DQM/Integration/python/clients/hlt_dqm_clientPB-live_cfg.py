import FWCore.ParameterSet.Config as cms

process = cms.Process("HARVESTING")

#----------------------------
#### Histograms Source
#----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.pbsource_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'HLTpb'
process.dqmEnv.eventInfoFolder = 'EventInfo'
process.dqmSaver.tag = 'HLTpb'
#process.dqmSaver.path = './HLT'
#-----------------------------

# customise for playback
if process.dqmRunConfig.type.value() is "playback":
    process.dqmEnv.eventInfoFolder = 'EventInfo/Random'

# DQM Modules
# FastTimerService client
process.load('HLTrigger.Timer.fastTimerServiceClient_cfi')
process.fastTimerServiceClient.dqmPath = "HLT/TimerService"

process.p = cms.EndPath( process.fastTimerServiceClient + process.dqmEnv + process.dqmSaver )
