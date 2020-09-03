from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# Define here the BeamSpotOnline record name,
# it will be used both in FakeBeamMonitor setup and in payload creation/upload
BSOnlineRecordName = 'BeamSpotOnlineHLTObjectsRcd'

#from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
#process = cms.Process("BeamMonitor", Run2_2018) # FIMXE
import sys
from Configuration.Eras.Era_Run2_2018_pp_on_AA_cff import Run2_2018_pp_on_AA
process = cms.Process("FakeBeamMonitor", Run2_2018_pp_on_AA)

# Message logger
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('*'),
#    cerr = cms.untracked.PSet(
#        FwkReport = cms.untracked.PSet(
#            optionalPSet = cms.untracked.bool(True),
#            reportEvery = cms.untracked.int32(1000),
#            limit = cms.untracked.int32(999999)
#        )
#    ),
#    destinations = cms.untracked.vstring('cerr'),
#)

unitTest=False
if 'unitTest=True' in sys.argv:
  unitTest=True

# Switch to veto the upload of the BeamSpot conditions to the DB
# when False it performs the upload
noDB = True
if 'noDB=False' in sys.argv:
    noDB=False

# Common part for PP and H.I Running
#-----------------------------
if unitTest:
  process.load("DQM.Integration.config.unittestinputsource_cfi")
  from DQM.Integration.config.unittestinputsource_cfi import options
else:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options

  # new stream label
  process.source.streamLabel = cms.untracked.string('streamDQMOnlineBeamspot')

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

#-----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.FakeBeamMonitor_cff")

#---------------
# Calibration
#---------------
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

# Change Beam Monitor variables
#if process.dqmRunConfig.type.value() is "production":
#  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
#  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
#  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
#  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt'
#else:
#  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
#  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
#  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
#  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmdev/BeamMonitorDQM/BeamFitResults.txt'

process.dqmcommon = cms.Sequence(process.dqmEnv
                               * process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamMonitor)

#-----------------------------------------------------------
# process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

#--------------------------
# Proton-Proton Stuff
#--------------------------

if (process.runType.getRunType() == process.runType.pp_run or
    process.runType.getRunType() == process.runType.pp_run_stage1 or
    process.runType.getRunType() == process.runType.cosmic_run or
    process.runType.getRunType() == process.runType.cosmic_run_stage1 or 
    process.runType.getRunType() == process.runType.hpu_run or
    process.runType.getRunType() == process.runType.hi_run):

    print("[beamhlt_dqm_sourceclient-live_cfg]:: Running pp")

    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

    process.dqmBeamMonitor.monitorName = 'TrackingHLTBeamspotStream'

    process.dqmBeamMonitor.OnlineMode = True              
    process.dqmBeamMonitor.recordName = BSOnlineRecordName

    process.dqmBeamMonitor.resetEveryNLumi   = 5
    process.dqmBeamMonitor.resetPVEveryNLumi = 5

    #process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
    #process.dqmBeamMonitor.PVFitter.minVertexNdf        = 10
  
    # some inputs to BeamMonitor
    #if(process.runType.getRunType() == process.runType.hi_run):
    #  process.dqmBeamMonitor.BeamFitter.TrackCollection = 'hltPFMuonMergingPPOnAA'
    #  process.dqmBeamMonitor.primaryVertex              = 'hltVerticesPFFilterPPOnAA'
    #  process.dqmBeamMonitor.PVFitter.VertexCollection  = 'hltVerticesPFFilterPPOnAA'
    #else:
    #  process.dqmBeamMonitor.BeamFitter.TrackCollection = 'hltPFMuonMerging'
    #  process.dqmBeamMonitor.primaryVertex              = 'hltVerticesPFFilter'
    #  process.dqmBeamMonitor.PVFitter.VertexCollection  = 'hltVerticesPFFilter'

    # keep checking this with new release expected close to 1
    #process.dqmBeamMonitor.PVFitter.errorScale = 0.95

    #TriggerName for selecting pv for DIP publication, NO wildcard needed here
    #it will pick all triggers which has these strings in theri name
    #process.dqmBeamMonitor.jetTrigger = cms.untracked.vstring(
    #    "HLT_HT300_Beamspot", "HLT_HT300_Beamspot",
    #    "HLT_PAZeroBias_v", "HLT_ZeroBias_", "HLT_QuadJet",
    #    "HLT_HI")

    #process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")

    #---------
    # Upload BeamSpotOnlineObject (HLTRcd) to CondDB
    process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",

        DBParameters = cms.PSet(
                                messageLevel = cms.untracked.int32(0),
                                authenticationPath = cms.untracked.string('')
                               ),

        # Upload to CondDB
        connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS'),
        preLoadConnectionString = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS'),

        runNumber = cms.untracked.uint64(options.runNumber),
        lastLumiFile = cms.untracked.string('last_lumi.txt'),
        #lastLumiUrl = cms.untracked.string('http://ru-c2e14-11-01.cms:11100/urn:xdaq-application:lid=52/getLatestLumiSection'),
        writeTransactionDelay = cms.untracked.uint32(options.transDelay),
        autoCommit = cms.untracked.bool(True),
        toPut = cms.VPSet(cms.PSet(
            record = cms.string(BSOnlineRecordName),
            tag = cms.string('BeamSpotOnlineTestHLT'),
            timetype = cms.untracked.string('Lumi'),
            onlyAppendUpdatePolicy = cms.untracked.bool(True)
        ))
    )

    # If not live or noDB: produce a (local) SQLITE file
    if unitTest or noDB:
      process.OnlineDBOutputService.connect = cms.string('sqlite_file:BeamSpotOnlineHLT.db')
      process.OnlineDBOutputService.preLoadConnectionString = cms.untracked.string('sqlite_file:BeamSpotOnlineHLT.db')

    process.p = cms.Path(process.dqmcommon
                        * process.monitor )
