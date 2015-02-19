import FWCore.ParameterSet.Config as cms

process = cms.Process("HARVESTING")

#----------------------------
#### Histograms Source
#----------------------------
# for live online DQM in P5
process.load("DQM.Integration.test.pbsource_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'HLT'
process.dqmEnv.eventInfoFolder = 'EventInfo/HLTfromPBfile'
process.dqmSaver.dirName = './HLT'
#-----------------------------

# DQM Modules
# FastTimerService client
process.load('HLTrigger.Timer.fastTimerServiceClient_cfi')
process.fastTimerServiceClient.dqmPath = "HLT/TimerService"

process.p = cms.EndPath( process.fastTimerServiceClient + process.dqmEnv + process.dqmSaver )
