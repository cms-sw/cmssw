import FWCore.ParameterSet.Config as cms
from DQM.HcalMonitorTasks.HcalMonitorTasks_cfi import SetTaskParams

import os, sys, socket, string

process = cms.Process("HCALDQM")
subsystem="HcalCalib"

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = subsystem
# process.dqmSaver.path= "."
process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/hcal_reference.root'

print "Running with run type = ", process.runType.getRunType()

# Set this to True is running in Heavy Ion mode
HEAVYION=False
if process.runType.getRunType() == process.runType.hi_run:
      HEAVYION=True

# Get Host information
host = socket.gethostname().split('.')[0].lower()
HcalPlaybackHost='dqm-c2d07-13'.lower()
HcalCalibPlaybackHost='dqm-c2d07-16'.lower()
#HcalPlaybackHost='srv-c2d04-25'.lower()
#HcalCalibPlaybackHost='srv-c2d04-28'.lower()


playbackHCALCALIB=False
if (host==HcalCalibPlaybackHost):
    playbackHCALCALIB=True

#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5
#process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
process.load("DQM.Integration.config.fileinputsource_cfi")

#process.DQMEventStreamHttpReader.consumerName = 'Hcal Orbit Gap DQM Consumer'
#process.DQMEventStreamHttpReader.SelectEvents =  cms.untracked.PSet(SelectEvents = cms.vstring('HLT_HcalCalibratio*','HLT_TechTrigHCALNoise*','HLT_L1Tech_HBHEHO_totalOR*'))
#process.DQMEventStreamHttpReader.max_queue_depth = cms.int32(100)
#process.DQMEventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputCalibration')
#if (HEAVYION):
#    process.DQMEventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputCalibrationHI')
#    process.DQMEventStreamHttpReader.SelectEvents =  cms.untracked.PSet(SelectEvents = cms.vstring('HLT_HIHcalCalibration_v*','HLT_HcalCalibration_v*'))
    
#process.DQMEventStreamHttpReader.sourceURL = cms.string('http://%s:23100/urn:xdaq-application:lid=30' % socket.gethostname())
#-----------------------------
# Hcal Conditions: from Global Conditions Tag 
#-----------------------------
# DB Condition for online cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# DB condition for offline test
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

process.load("FWCore.MessageLogger.MessageLogger_cfi")


#-----------------------------
# Hcal DQM Source, including SimpleReconstrctor
#-----------------------------
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi")

# Only use this correction in CMSSW_3_9_1 and above, after hbhereco was renamed!
#print process.hbheprereco

version=os.getenv("CMSSW_VERSION").split("_")
version1=string.atoi(version[1])
version2=string.atoi(version[2])

# Use prereco for all releases >= 3_9_X
if (version1>3) or (version1==3 and version2>=9):
    process.hbhereco = process.hbheprereco.clone()
        
# Allow all rechits in mark&pass events
process.hfreco.dropZSmarkedPassed=False
process.horeco.dropZSmarkedPassed=False
process.hbhereco.dropZSmarkedPassed=False


# Turn off default blocking of dead/off channels from rechit reconstructor
process.essourceSev =  cms.ESSource("EmptyESSource",
                                               recordName = cms.string("HcalSeverityLevelComputerRcd"),
                                               firstValid = cms.vuint32(1),
                                               iovIsRunNotTime = cms.bool(True)
                            )

process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
process.hcalRecAlgos.DropChannelStatusBits = cms.vstring('') # Had been ('HcalCellOff','HcalCellDead')


# -------------------------------
# Hcal DQM Modules
# -------------------------------

process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
process.load("DQM.HcalMonitorTasks.HcalMonitorTasks_cfi")
# Set individual parameters for the tasks
process.load("DQM.HcalMonitorTasks.HcalCalibTasksOnline_cff")
process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")

process.hcalDetDiagLaserMonitor.LaserReferenceData       = '/dqmdata/dqm/reference/hcalcalib_laser_reference.root'
process.hcalDetDiagPedestalMonitor.PedestalReferenceData = '/dqmdata/dqm/reference/hcalcalib_pedestal_reference.root'

# As of 23 March 2010, cannot write extra root/html files from online DQM!
process.hcalDetDiagLaserMonitor.OutputFilePath           = '/nfshome0/hcaldqm/DQM_OUTPUT/DetDiag/DetDiagDatasets_Temp/'
process.hcalDetDiagPedestalMonitor.OutputFilePath        = '/nfshome0/hcaldqm/DQM_OUTPUT/DetDiag/DetDiagDatasets_Temp/'
process.hcalDetDiagNoiseMonitor.OutputFilePath        = '/nfshome0/hcaldqm/DQM_OUTPUT/DetDiag/DetDiagDatasets_Temp/'

# disable output from playback server
if playbackHCALCALIB==True:
    process.hcalDetDiagLaserMonitor.OutputFilePath=''
    process.hcalDetDiagPedestalMonitor.OutputFilePath =''

# Set all directories to HcalCalib/
if not subsystem.endswith("/"):
    subsystem=subsystem+"/"
process.hcalMonitor.subSystemFolder=subsystem
SetTaskParams(process,"subSystemFolder",subsystem)
process.hcalClient.subSystemFolder=subsystem

# special (hopefully temporary) parameter to fix crash in endJob of HcalDQM
# 18 April 2011:  Only enable the following line if hcalClient is crashing:
#process.hcalClient.online=True


#-----------------------------
# Hcal DQM Client
#-----------------------------

# hcalClient configurable values ------------------------
# suppresses html output from HCalClient  

# As of 23 March 2010, cannot write extra html output files from online DQM!
process.hcalClient.baseHtmlDir = ''   #'/nfshome0/hcaldqm/DQM_OUTPUT/DetDiag/DetDiag_HTML/'  # set to '' to prevent html output
#process.hcalClient.htmlUpdateTime=120  # update every two hours
#process.hcalClient.htmlFirstUpdate=20  # start after 20 minutes

process.hcalClient.RawData_minerrorrate = cms.untracked.double(2.) # ignore errors from dataformat client

# Don't create problem histograms for tasks that aren't run:
process.hcalClient.enabledClients = [#"DeadCellMonitor",
                                     #"HotCellMonitor",
                                     "RecHitMonitor",
                                     #"DigiMonitor",
                                     "RawDataMonitor",
                                     #"TrigPrimMonitor",
                                     #"NZSMonitor",
                                     #"BeamMonitor",
                                     "DetDiagPedestalMonitor",
                                     "DetDiagLaserMonitor",
                                     #"DetDiagLEDMonitor",
                                     "DetDiagNoiseMonitor",
                                     "DetDiagTimingMonitor",
                                     "Summary"]


# ----------------------
# Trigger Unpacker Stuff
# ----------------------
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("L1Trigger.Configuration.L1DummyConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.l1GtUnpack.DaqGtInputTag = 'source'

#-----------------------------
# Scheduling
#-----------------------------
process.options = cms.untracked.PSet(
  SkipEvent = cms.untracked.vstring('ProductNotFound')
)


#################################################################
#                                                               #
#  THE FOLLOWING CHANGES ARE NEEDED FOR HEAVY ION RUNNING       #
#                                                               #
#################################################################

#if (HEAVYION):
    # Define new Heavy Ion labels
## The section below is commented out due to being obsolete in 2011
#    process.hcalDigis.InputLabel                    = cms.InputTag("hltHcalCalibrationRaw")
#    process.hcalDetDiagNoiseMonitor.RawDataLabel    = cms.untracked.InputTag("hltHcalCalibrationRaw")
#    process.hcalDetDiagLaserMonitor.RawDataLabel    = cms.untracked.InputTag("hltHcalCalibrationRaw")
#    process.hcalDetDiagPedestalMonitor.rawDataLabel = cms.untracked.InputTag("hltHcalCalibrationRaw")
#    process.hcalMonitor.FEDRawDataCollection        = cms.untracked.InputTag("hltHcalCalibrationRaw")
#    process.hcalRawDataMonitor.FEDRawDataCollection = cms.untracked.InputTag("hltHcalCalibrationRaw")
    #process.hcalRawDataMonitor.digiLabel            = cms.untracked.InputTag("hltHcalCalibrationRaw")

#    process.hltHighLevel = cms.EDFilter("HLTHighLevel",
#                                        TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),  
#                                        HLTPaths = cms.vstring('HLT_HcalCalibration_HI'),        # provide list of HLT paths (or patterns) you want
#                                        eventSetupPathsKey = cms.string(''),             # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
#                                        andOr = cms.bool(True),                          # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
#                                        throw = cms.bool(True)                           # throw exception on unknown path names
#                                        )

#    process.filterSequence = cms.Sequence(
#        process.hltHighLevel
#        )

#    process.p = cms.Path(process.hcalDigis
#                         *process.l1GtUnpack
#                         *process.horeco
#                         *process.hfreco
#                         *process.hbhereco
#                         *process.hcalMonitor
#                         *process.hcalMonitorTasksCalibrationSequence 
#                         *process.hcalClient
#                         *process.dqmEnv
#                         *process.dqmSaver)

#################################################################




process.p = cms.Path(process.hcalDigis
                         #*process.l1GtUnpack
                         *process.horeco
                         *process.hfreco
                         *process.hbhereco
                         *process.hcalMonitor
                         *process.hcalMonitorTasksCalibrationSequence 
                         *process.hcalClient
                         *process.dqmEnv
                         *process.dqmSaver)


#-----------------------------
# Quality Tester 
# will add switch to select histograms to be saved soon
#-----------------------------
process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/HcalMonitorClient/data/hcal_qualitytest_config.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)

process.hcalDigis.InputLabel = cms.InputTag("hltHcalCalibrationRaw")
process.l1GtUnpack.DaqGtInputTag = cms.InputTag("hltHcalCalibrationRaw")
process.hcalDataIntegrityMonitor.RawDataLabel = cms.untracked.InputTag("hltHcalCalibrationRaw")
process.hcalDetDiagNoiseMonitor.RawDataLabel = cms.untracked.InputTag("hltHcalCalibrationRaw")
process.hcalDetDiagPedestalMonitor.rawDataLabel = cms.untracked.InputTag("hltHcalCalibrationRaw")
process.hcalDetDiagTimingMonitor.FEDRawDataCollection = cms.untracked.InputTag("hltHcalCalibrationRaw")
process.hcalMonitor.FEDRawDataCollection = cms.untracked.InputTag("hltHcalCalibrationRaw")
process.hcalNZSMonitor.RawDataLabel = cms.untracked.InputTag("hltHcalCalibrationRaw")
process.hcalNoiseMonitor.RawDataLabel = cms.untracked.InputTag("hltHcalCalibrationRaw")
process.hcalDetDiagLaserMonitor.RawDataLabel = cms.untracked.InputTag("hltHcalCalibrationRaw")
process.hcalRawDataMonitor.FEDRawDataCollection = cms.untracked.InputTag("hltHcalCalibrationRaw")
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------
if (HEAVYION):
     process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
     process.l1GtUnpack.DaqGtInputTag = cms.InputTag("rawDataRepacker")
     process.hcalDataIntegrityMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
     process.hcalDetDiagNoiseMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
     process.hcalDetDiagPedestalMonitor.rawDataLabel = cms.untracked.InputTag("rawDataRepacker")
     process.hcalDetDiagTimingMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataRepacker")
     process.hcalMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataRepacker")
     process.hcalNZSMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
     process.hcalNoiseMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
     process.hcalDetDiagLaserMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
     process.hcalRawDataMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataRepacker")


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
