from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# Define here the BeamSpotOnline record name,
# it will be used both in FakeBeamMonitor setup and in payload creation/upload
BSOnlineRecordName = 'BeamSpotOnlineHLTObjectsRcd'

import sys
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("FakeBeamMonitor", Run2_2018)

# switch
live = True # FIXME
unitTest = False

if 'unitTest=True' in sys.argv:
  live=False
  unitTest=True


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
process.dqmEnv.subSystemFolder = 'FakeBeamMonitor'
process.dqmSaver.tag           = 'FakeBeamMonitor'
process.dqmSaver.runNumber     = options.runNumber
process.dqmSaverPB.tag         = 'FakeBeamMonitor'
process.dqmSaverPB.runNumber   = options.runNumber

#---------------
"""
# Conditions
if (live):
  process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
else:
  process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
  from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
  process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')
  # you may need to set manually the GT in the line below
  #process.GlobalTag.globaltag = '100X_upgrade2018_realistic_v10'
"""
#-----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.FakeBeamMonitor_cff")
process.dqmBeamMonitor = process.dqmFakeBeamMonitor.clone()

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
  rawDataInputTag = cms.InputTag("rawDataRepacker")
else:
  rawDataInputTag = cms.InputTag("rawDataCollector")

process.dqmBeamMonitor.monitorName = 'FakeBeamMonitor'
process.dqmBeamMonitor.OnlineMode = True              
process.dqmBeamMonitor.recordName = BSOnlineRecordName

process.dqmBeamMonitor.resetEveryNLumi   = 5
process.dqmBeamMonitor.resetPVEveryNLumi = 5

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
      #lastLumiFile = cms.untracked.string('last_lumi.txt'),
      lastLumiUrl = cms.untracked.string('http://ru-c2e14-11-01.cms:11100/urn:xdaq-application:lid=52/getLatestLumiSection'),
      writeTransactionDelay = cms.untracked.uint32(options.transDelay),
      latency = cms.untracked.uint32(2),
      autoCommit = cms.untracked.bool(True),
      saveLogsOnDB = cms.untracked.bool(True),
      jobName = cms.untracked.string("BeamSpotOnlineHLTTest"), # name of the DB log record
      toPut = cms.VPSet(cms.PSet(
          record = cms.string(BSOnlineRecordName),
          tag = cms.string('BeamSpotOnlineTestHLT'),
          timetype = cms.untracked.string('Lumi'),
          onlyAppendUpdatePolicy = cms.untracked.bool(True)
      ))
  )

else:
  process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",
    DBParameters = cms.PSet(
                            messageLevel = cms.untracked.int32(0),
                            authenticationPath = cms.untracked.string('.')
                            ),

    # Upload to CondDB
    connect = cms.string('sqlite_file:BeamSpotOnlineHLT.db'),
    preLoadConnectionString = cms.untracked.string('sqlite_file:BeamSpotOnlineHLT.db'),

    runNumber = cms.untracked.uint64(options.runNumber),
    lastLumiFile = cms.untracked.string('last_lumi.txt'),
    #lastLumiUrl = cms.untracked.string('http://ru-c2e14-11-01.cms:11100/urn:xdaq-application:lid=52/getLatestLumiSection'),
    writeTransactionDelay = cms.untracked.uint32(options.transDelay),
    latency = cms.untracked.uint32(2),
    autoCommit = cms.untracked.bool(True),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string(BSOnlineRecordName),
        tag = cms.string('BeamSpotOnlineTestHLT'),
        timetype = cms.untracked.string('Lumi'),
        onlyAppendUpdatePolicy = cms.untracked.bool(True)
    ))
)

process.p = cms.Path(process.dqmcommon
                    * process.monitor )
