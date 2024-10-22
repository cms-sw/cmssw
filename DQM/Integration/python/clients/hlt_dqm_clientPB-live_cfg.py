import FWCore.ParameterSet.Config as cms
import sys

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("HARVESTING", Run3)

unitTest = False
if 'unitTest=True' in sys.argv:
	unitTest=True

#----------------------------
#### Histograms Source
#----------------------------

if unitTest:
   process.load("DQM.Integration.config.unittestinputsource_cfi")
   from DQM.Integration.config.unittestinputsource_cfi import options
else:
   # for live online DQM in P5
   process.load("DQM.Integration.config.pbsource_cfi")
   from DQM.Integration.config.pbsource_cfi import options

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'HLTpb'
process.dqmEnv.eventInfoFolder = 'EventInfo'
process.dqmSaver.tag = 'HLTpb'
#process.dqmSaver.path = './HLT'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'HLTpb'
process.dqmSaverPB.runNumber = options.runNumber
#-----------------------------

# customise for playback
if process.dqmRunConfig.type.value() == "playback":
    process.dqmEnv.eventInfoFolder = 'EventInfo/Random'

# DQM Modules
# FastTimerService client
process.load('HLTrigger.Timer.fastTimerServiceClient_cfi')
process.fastTimerServiceClient.dqmPath = "HLT/TimerService"
# timing VS lumi
process.fastTimerServiceClient.doPlotsVsOnlineLumi = True
process.fastTimerServiceClient.doPlotsVsPixelLumi = False
process.fastTimerServiceClient.onlineLumiME = dict(
    folder = 'HLT/LumiMonitoring',
    name   = 'lumiVsLS',
    nbins  = 6000,
    xmin   = 0,
    xmax   = 30000,
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
      nbinsX = cms.int32( 6000 ),
      xminX  = cms.double( 0. ),
      xmaxX  = cms.double( 30000. ),
      doYaxis = cms.bool( False ),
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

print("Final Source settings:", process.source)
process.p = cms.EndPath( process.fastTimerServiceClient + process.throughputServiceClient + process.psColumnVsLumi + process.dqmEnv + process.dqmSaver + process.dqmSaverPB )
