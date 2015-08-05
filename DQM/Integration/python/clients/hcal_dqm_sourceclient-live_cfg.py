import FWCore.ParameterSet.Config as cms

import os, sys, socket, string
from DQM.HcalMonitorTasks.HcalMonitorTasks_cfi import SetTaskParams


process = cms.Process("HCALDQM")
subsystem="Hcal" # specify subsystem name here

#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = subsystem
process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/hcal_reference.root'

print "Running with run type = ", process.runType.getRunType()

# Set this to True if running in Heavy Ion mode
HEAVYION=False
if process.runType.getRunType() == process.runType.hi_run:
  HEAVYION=True
 
# Get Host information
host = socket.gethostname().split('.')[0].lower()
HcalPlaybackHost='dqm-c2d07-13'.lower()
HcalCalibPlaybackHost='dqm-c2d07-16'.lower()
# These are playback servers, not hosts...
#HcalPlaybackHost='srv-c2d04-25'.lower()
#HcalCalibPlaybackHost='srv-c2d04-28'.lower()

playbackHCAL=False
if (host==HcalPlaybackHost):
    playbackHCAL=True

#-----------------------------
# Hcal Conditions: from Global Conditions Tag 
#-----------------------------
# DB Condition for online cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# DB condition for offline test
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")


#-----------------------------
# Hcal DQM Source, including Rec Hit Reconstructor
#-----------------------------
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")

# Only use this correction in CMSSW_3_9_1 and above, after hbhereco was renamed!
#print process.hbheprereco

version=os.getenv("CMSSW_VERSION").split("_")
version1=string.atoi(version[1])
version2=string.atoi(version[2])

# Use prereco for all releases >= 3_9_X
if (version1>3) or (version1==3 and version2>=9):
    process.hbhereco = process.hbheprereco.clone()

# Turn off default blocking of dead channels from rechit collection
process.essourceSev =  cms.ESSource("EmptyESSource",
                                    recordName = cms.string("HcalSeverityLevelComputerRcd"),
                                    firstValid = cms.vuint32(1),
                                    iovIsRunNotTime = cms.bool(True)
                                    )
process.hcalRecAlgos.DropChannelStatusBits = cms.vstring('') # Had been ('HcalCellOff','HcalCellDead')

#----------------------------
# Trigger Emulator
#----------------------------
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.valHcalTriggerPrimitiveDigis = process.simHcalTriggerPrimitiveDigis.clone()
process.valHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag('hcalDigis', 'hcalDigis')
process.valHcalTriggerPrimitiveDigis.FrontEndFormatError = cms.bool(True)

#configuration used in Heavy Ion runs only
if (HEAVYION):
    process.valHcalTriggerPrimitiveDigis.FrontEndFormatError = cms.bool(False) 
    #process.hcalDeadCellMonitor.minDeadEventCount = 10

process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)
process.valHcalTriggerPrimitiveDigis.FG_threshold = cms.uint32(2)
process.valHcalTriggerPrimitiveDigis.InputTagFEDRaw = cms.InputTag("rawDataCollector")

# -------------------------------
# Hcal DQM Modules
# -------------------------------

process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
#process.load("DQM.HcalMonitorModule.ZDCMonitorModule_cfi")

process.load("DQM.HcalMonitorTasks.HcalMonitorTasks_cfi")
# Set individual parameters for the tasks
process.load("DQM.HcalMonitorTasks.HcalTasksOnline_cff")
process.hcalBeamMonitor.lumiqualitydir="/nfshome0/hcaldqm/DQM_OUTPUT/lumi/"
if playbackHCAL==True:
    process.hcalBeamMonitor.lumiqualitydir="/nfshome0/hcaldqm/DQM_OUTPUT/lumi_playback/"


process.hcalBeamMonitor.hotrate=0.40

process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")
#process.load("DQM.HcalMonitorTasks.HcalZDCMonitor_cfi")

#-----------------------------
#  Configure Hcal DQM
#-----------------------------
# Our subsystem values expected a '/' at end
# Source code should catch when it's not there, but don't take the chance yet:
if not subsystem.endswith("/"):
    subsystem=subsystem+"/"
process.hcalMonitor.subSystemFolder=subsystem
SetTaskParams(process,"subSystemFolder",subsystem)
process.hcalClient.subSystemFolder=subsystem

# special (hopefully temporary) parameter to fix crash in endJob of HcalDQM
# 18 April 2011:  Only enable the next line if hcalClient is crashing:
#process.hcalClient.online=True

#print "BITS = ",process.hcalRecHitMonitor.HcalHLTBits.value()
if (HEAVYION):
    process.hcalRecHitMonitor.HcalHLTBits=["HLT_HIActivityHF_Coincidence3",
                                           "HLT_HIL1Tech_HCAL_HF"]
    
    process.hcalRecHitMonitor.MinBiasHLTBits=["HLT_HIMinBiasBSC",
                                              "HLT_HIL1Tech_BSC_minBias"
                                              ]

else:
    process.hcalRecHitMonitor.HcalHLTBits=["HLT_L1Tech_HCAL_HF",
                                           "HLT_L1Tech_BSC_minBias_treshold1"]
    
    process.hcalRecHitMonitor.MinBiasHLTBits=["HLT_MinBiasPixel_SingleTrack",
                                              "HLT_L1Tech_BSC_minBias",
                                              "HLT_L1Tech_BSC_minBias_OR",
                                              "HLT_L1Tech_BSC_minBias_threshold1",
                                              "HLT_ZeroBias"
                                              ]
    
    process.hcalDigiMonitor.MinBiasHLTBits=["HLT_MinBiasPixel_SingleTrack",
                                            "HLT_L1Tech_BSC_minBias",
                                            "HLT_L1Tech_BSC_minBias_OR",
                                            "HLT_L1Tech_BSC_minBias_threshold1",
                                            "HLT_ZeroBias"
                                            ]


#print "NEW BITS = ",process.hcalRecHitMonitor.HcalHLTBits.value()

# hcalClient configurable values ------------------------
# suppresses html output from HCalClient  
process.hcalClient.baseHtmlDir = ''  # set to '' to prevent html output

# Update once per hour, starting after 10 minutes
process.hcalClient.databaseDir = '/nfshome0/hcaldqm/DQM_OUTPUT/ChannelStatus/' # set to empty to suppress channel status output

if (playbackHCAL==True):
    process.hcalClient.databaseDir = ''
process.hcalClient.databaseFirstUpdate=10
process.hcalClient.databaseUpdateTime=60

# Set values higher at startup  (set back from 0.25 to 0.05 on 15 April 2010)
process.hcalClient.DeadCell_minerrorrate=0.05
process.hcalClient.HotCell_minerrorrate =cms.untracked.double(0.10)

# Disable the HFLumi from affecting HF Summary Value
process.hcalClient.Beam_minerrorrate=cms.untracked.double(2.0)

process.hcalDigiMonitor.maxDigiSizeHF = cms.untracked.int32(10)
# Increase hotcellmonitor thresholds for HI runs
if (HEAVYION):
    process.hcalHotCellMonitor.ETThreshold = cms.untracked.double(10.0)
    process.hcalHotCellMonitor.ETThreshold_HF  = cms.untracked.double(10.0)
    
if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
    process.hcalDetDiagTimingMonitor.CosmicsCorr=True


# Don't create problem histograms for tasks that aren't run:
process.hcalClient.enabledClients = ["DeadCellMonitor",
                                     "HotCellMonitor",
                                     "RecHitMonitor",
                                     "DigiMonitor",
                                     "RawDataMonitor",
                                     "TrigPrimMonitor",
                                     "NZSMonitor",
                                     "BeamMonitor",
#                                     "ZDCMonitor",
                                     #"DetDiagPedestalMonitor",
                                     #"DetDiagLaserMonitor",
                                     #"DetDiagLEDMonitor",
                                     #"DetDiagNoiseMonitor",
                                     "CoarsePedestalMonitor",
                                     "DetDiagTimingMonitor",
                                     "Summary"
                                     ]


# Set expected idle BCN time to correct value
#(6 for runs < 116401; 3560 for runs > c. 117900, 3563 for runs between)
# (3559 starting on 7 December 2009)

idle=3559
process.hcalDigis.ExpectedOrbitMessageTime=cms.untracked.int32(idle)
process.hcalDigiMonitor.ExpectedOrbitMessageTime = idle
process.hcalDigiMonitor.shutOffOrbitTest=False

# Turn off dead cell checks in HO ring 2
process.hcalDeadCellMonitor.excludeHORing2 = False

# Ignore ped-ref differences
process.hcalCoarsePedestalMonitor.ADCDiffThresh = 2
# block both hot and dead channels from CoarsePedestal Monitor
process.hcalClient.CoarsePedestal_BadChannelStatusMask=cms.untracked.int32((1<<5) | (1<<6))

# Allow even bad-quality digis
#process.hcalDigis.FilterDataQuality=False

# ----------------------
# Trigger Unpacker Stuff
# ----------------------
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("L1Trigger.Configuration.L1DummyConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.load('Configuration/StandardSequences/RawToDigi_Data_cff') #to unpack l1gtEvm
process.gtEvmDigis.UnpackBxInEvent = cms.int32(1) #to unpack l1gtEvm
process.l1GtUnpack.DaqGtInputTag = 'source'


#-----------------------------
# Scheduling
#-----------------------------
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)

#-----------------------------
# Quality Tester 
# check Dead Cells VS LS for RBX losses
#-----------------------------
process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/HcalMonitorClient/data/hcal_qualitytest_config.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)

process.p = cms.Path(process.hcalDigis
                     *process.valHcalTriggerPrimitiveDigis
                     #*process.gtEvmDigis#to unpack l1gtEvm
                     *process.l1GtUnpack
                     *process.horeco
                     *process.hfreco
                     *process.hbhereco
                     *process.zdcreco
                     *process.hcalMonitor
                     *process.hcalMonitorTasksOnlineSequence 
                     *process.hcalClient
                     *process.qTester
                     #*process.hcalZDCMonitor
                     *process.dqmEnv
                     *process.dqmSaver)



process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.csctfDigis.producer = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.l1GtUnpack.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")
process.hcalDataIntegrityMonitor.RawDataLabel = cms.untracked.InputTag("rawDataCollector")
process.hcalDetDiagNoiseMonitor.RawDataLabel = cms.untracked.InputTag("rawDataCollector")
process.hcalDetDiagPedestalMonitor.rawDataLabel = cms.untracked.InputTag("rawDataCollector")
process.hcalDetDiagTimingMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataCollector")
process.hcalMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataCollector")
process.hcalNZSMonitor.RawDataLabel = cms.untracked.InputTag("rawDataCollector")
process.hcalNoiseMonitor.RawDataLabel = cms.untracked.InputTag("rawDataCollector")
process.hcalRawDataMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataCollector")
#process.zdcMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataCollector")

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

if (HEAVYION):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.l1GtUnpack.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.hcalDataIntegrityMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
    process.hcalDetDiagNoiseMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
    process.hcalDetDiagPedestalMonitor.rawDataLabel = cms.untracked.InputTag("rawDataRepacker")
    process.hcalDetDiagTimingMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataRepacker")
    process.hcalMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataRepacker")
    process.hcalNZSMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
    process.hcalNoiseMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
    process.hcalRawDataMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataRepacker")
    process.hcalDigiMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataRepacker")
#    process.zdcMonitor.FEDRawDataCollection = cms.untracked.InputTag("rawDataRepacker")


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
