from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# Define here the BeamSpotOnline record name,
# it will be used both in FakeBeamMonitor setup and in payload creation/upload
BSOnlineRecordName = 'BeamSpotOnlineHLTObjectsRcd'
BSOnlineTag = 'BeamSpotOnlineFakeHLT'
BSOnlineJobName = 'BeamSpotOnlineFakeHLT'
BSOnlineOmsServiceUrl = 'http://cmsoms-eventing.cms:9949/urn:xdaq-application:lid=100/getRunAndLumiSection'
useLockRecords = True

import sys
if 'runkey=hi_run' in sys.argv:
  from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_cff import Run3_pp_on_PbPb_approxSiStripClusters
  process = cms.Process("FakeBeamMonitorHLT", Run3_pp_on_PbPb_approxSiStripClusters)
else:
  from Configuration.Eras.Era_Run3_cff import Run3
  process = cms.Process("FakeBeamMonitorHLT", Run3)

# switch
live = True # FIXME
unitTest = False

if 'unitTest=True' in sys.argv:
  live=False
  unitTest=True
  useLockRecords = False

# Common part for PP and H.I Running
#-----------------------------
if unitTest:
  process.load("DQM.Integration.config.unittestinputsource_cfi")
  from DQM.Integration.config.unittestinputsource_cfi import options
elif live:
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options
else:
  process.load("DQM.Integration.config.fileinputsource_cfi")
  from DQM.Integration.config.fileinputsource_cfi import options

# new stream label
#process.source.streamLabel = cms.untracked.string('streamDQMOnlineBeamspot')

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

#--------------------------
# HLT Filter
# 0=random, 1=physics, 2=calibration, 3=technical
#--------------------------
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)

#-----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'FakeBeamMonitorHLT'
process.dqmSaver.tag           = 'FakeBeamMonitorHLT'
process.dqmSaver.runNumber     = options.runNumber
process.dqmSaverPB.tag         = 'FakeBeamMonitorHLT'
process.dqmSaverPB.runNumber   = options.runNumber

# Configure tag and jobName if running Playback system
if process.isDqmPlayback.value :
  BSOnlineTag = BSOnlineTag + 'Playback'
  BSOnlineJobName = BSOnlineJobName + 'Playback'
  BSOnlineOmsServiceUrl = ''
  useLockRecords = False

#---------------
"""
# Conditions
if (live):
  process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
else:
  process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
  from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
  process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')
  # you may need to set manually the GT in the line below
  #process.GlobalTag.globaltag = '100X_upgrade2018_realistic_v10'
"""
#-----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.FakeBeamMonitor_cff")
process.dqmBeamMonitor = process.dqmFakeBeamMonitor.clone(
  monitorName = 'FakeBeamMonitor',
  OnlineMode = True,
  recordName = BSOnlineRecordName,
  useLockRecords = useLockRecords,
  resetEveryNLumi   = 5,
  resetPVEveryNLumi = 5
)  

#---------------
# Calibration
#---------------
# Condition for P5 cluster
#process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
process.dqmcommon = cms.Sequence(process.dqmEnv
                               * process.dqmSaver * process.dqmSaverPB)

process.monitor = cms.Sequence(process.dqmBeamMonitor)

#-----------------------------------------------------------
# process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

# Set rawDataRepacker (HI and live) or rawDataCollector (for all the rest)
if (process.runType.getRunType() == process.runType.hi_run and live):
  rawDataInputTag = "rawDataRepacker"
else:
  rawDataInputTag = "rawDataCollector"

#---------
# Upload BeamSpotOnlineObject (HLTRcd) to CondDB
if unitTest == False:
  process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",

      DBParameters = cms.PSet(
                              messageLevel = cms.untracked.int32(0),
                              authenticationPath = cms.untracked.string('.')
                              ),

      # Upload to CondDB
      connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS'),
      preLoadConnectionString = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS'),
      runNumber = cms.untracked.uint64(options.runNumber),
      omsServiceUrl = cms.untracked.string(BSOnlineOmsServiceUrl),
      latency = cms.untracked.uint32(2),
      autoCommit = cms.untracked.bool(True),
      saveLogsOnDB = cms.untracked.bool(True),
      jobName = cms.untracked.string(BSOnlineJobName), # name of the DB log record
      toPut = cms.VPSet(cms.PSet(
          record = cms.string(BSOnlineRecordName),
          tag = cms.string(BSOnlineTag),
          timetype = cms.untracked.string('Lumi'),
          onlyAppendUpdatePolicy = cms.untracked.bool(True)
      )),
      frontierKey = cms.untracked.string(options.runUniqueKey)
  )

else:
  process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",
    DBParameters = cms.PSet(
                            messageLevel = cms.untracked.int32(0),
                            authenticationPath = cms.untracked.string('.')
                            ),

    # Upload to CondDB
    connect = cms.string('sqlite_file:BeamSpotOnlineFakeHLT.db'),
    preLoadConnectionString = cms.untracked.string('sqlite_file:BeamSpotOnlineFakeHLT.db'),

    runNumber = cms.untracked.uint64(options.runNumber),
    lastLumiFile = cms.untracked.string('last_lumi.txt'),
    latency = cms.untracked.uint32(2),
    autoCommit = cms.untracked.bool(True),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string(BSOnlineRecordName),
        tag = cms.string(BSOnlineTag),
        timetype = cms.untracked.string('Lumi'),
        onlyAppendUpdatePolicy = cms.untracked.bool(True)
    )),
    frontierKey = cms.untracked.string(options.runUniqueKey)
)
print("Configured frontierKey", options.runUniqueKey)

#---------
# Final path
print("Final Source settings:", process.source)

process.p = cms.Path(process.dqmcommon
                    * process.monitor )

