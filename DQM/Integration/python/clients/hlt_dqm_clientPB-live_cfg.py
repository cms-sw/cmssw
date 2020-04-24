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
# timing VS lumi
process.fastTimerServiceClient.doPlotsVsScalLumi  = cms.bool(True)
process.fastTimerServiceClient.doPlotsVsPixelLumi = cms.bool(False)
process.fastTimerServiceClient.scalLumiME = cms.PSet(
    folder = cms.string('HLT/LumiMonitoring'),
    name   = cms.string('lumiVsLS'),
    nbins  = cms.int32(5000),
    xmin   = cms.double(0),
    xmax   = cms.double(20000)
)

# ThroughputService client
process.load("HLTrigger.Timer.throughputServiceClient_cfi")
process.throughputServiceClient.dqmPath = "HLT/Throughput"

# PS column VS lumi
process.load('DQM.HLTEvF.dqmCorrelationClient_cfi')
process.psColumnVsLumi = process.dqmCorrelationClient.clone(
   me = cms.PSet(
      folder = cms.string("HLT/PSMonitoring"),
      name   = cms.string("psColumnVSlumi"),
      doXaxis = cms.bool( True ),
      nbinsX = cms.int32( 5000),
      xminX  = cms.double(    0.),
      xmaxX  = cms.double(20000.),
      doYaxis = cms.bool( False ),
      nbinsY = cms.int32 (   8),
      xminY  = cms.double(   0.),
      xmaxY  = cms.double(   8.),
   ),
   me1 = cms.PSet(
      folder   = cms.string("HLT/LumiMonitoring"),
      name     = cms.string("lumiVsLS"),
      profileX = cms.bool(True)
   ),
   me2 = cms.PSet(
      folder   = cms.string("HLT/PSMonitoring"),
      name     = cms.string("psColumnIndexVsLS"),
      profileX = cms.bool(True)
   ),
)

process.load('DQM.HLTEvF.psMonitorClient_cfi')
process.psChecker = process.psMonitorClient.clone()


process.p = cms.EndPath( process.fastTimerServiceClient + process.throughputServiceClient + process.psColumnVsLumi + process.psChecker + process.dqmEnv + process.dqmSaver )
