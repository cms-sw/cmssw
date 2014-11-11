import FWCore.ParameterSet.Config as cms

import os, sys, socket
from DQM.HcalMonitorTasks.HcalMonitorTasks_cfi import SetTaskParams

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
    
process = cms.Process("HCALDQM")
subsystem="Hcal" # specify subsystem name here

#----------------------------
# Event Source
#-----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'Hcal DQM Consumer'

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = subsystem
process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/hcal_reference.root'

#-----------------------------
# Hcal Conditions: from Global Conditions Tag 
#-----------------------------
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")


#-----------------------------
# Hcal DQM Source, including Rec Hit Reconstructor
#-----------------------------
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi")

# HF Corrections to reconstruction
process.hfreco.firstSample = 3
process.hfreco.samplesToAdd = 4

# ZDC Corrections to reco
process.zdcreco.firstSample  = 4
process.zdcreco.samplesToAdd = 3
process.zdcreco.recoMethod   = 2

# Turn off default blocking of dead channels from rechit collection
process.essourceSev =  cms.ESSource("EmptyESSource",
                                    recordName = cms.string("HcalSeverityLevelComputerRcd"),
                                    firstValid = cms.vuint32(1),
                                    iovIsRunNotTime = cms.bool(True)
                                    )
process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
process.hcalRecAlgos.DropChannelStatusBits = cms.vstring('') # Had been ('HcalCellOff','HcalCellDead')

#----------------------------
# Trigger Emulator
#----------------------------
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.valHcalTriggerPrimitiveDigis = process.simHcalTriggerPrimitiveDigis.clone()
process.valHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag('hcalDigis', 'hcalDigis')
process.valHcalTriggerPrimitiveDigis.FrontEndFormatError = cms.untracked.bool(True)
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)
process.valHcalTriggerPrimitiveDigis.FG_threshold = cms.uint32(2)

# -------------------------------
# Hcal DQM Modules
# -------------------------------

process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
process.load("DQM.HcalMonitorModule.ZDCMonitorModule_cfi")

process.load("DQM.HcalMonitorTasks.HcalMonitorTasks_cfi")
# Set individual parameters for the tasks
process.load("DQM.HcalMonitorTasks.HcalTasksOnline_cff")
process.hcalBeamMonitor.lumiqualitydir="/nfshome0/hcaldqm/DQM_OUTPUT/lumi/"
if playbackHCAL==True:
    process.hcalBeamMonitor.lumiqualitydir="/nfshome0/hcaldqm/DQM_OUTPUT/lumi_playback/"
    
process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")
process.load("DQM.HcalMonitorClient.ZDCMonitorClient_cfi")

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
#print "BITS = ",process.hcalRecHitMonitor.HcalHLTBits.value()
process.hcalRecHitMonitor.HcalHLTBits=["HLT_L1Tech_HCAL_HF_coincidence_PM",
                                       "HLT_L1Tech_HCAL_HF"]

process.hcalRecHitMonitor.MinBiasHLTBits=["HLT_MinBiasBSC",
                                          "HLT_L1Tech_BSC_minBias"
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
                   
# Don't create problem histograms for tasks that aren't run:
process.hcalClient.enabledClients = ["DeadCellMonitor",
                                     "HotCellMonitor",
                                     "RecHitMonitor",
                                     "DigiMonitor",
                                     "RawDataMonitor",
                                     "TrigPrimMonitor",
                                     "NZSMonitor",
                                     "BeamMonitor",
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
process.hcalDeadCellMonitor.excludeHORing2 = True

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
process.l1GtUnpack.DaqGtInputTag = 'source'


#-----------------------------
# Scheduling
#-----------------------------
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)

process.p = cms.Path(process.hcalDigis
                     *process.valHcalTriggerPrimitiveDigis
                     *process.l1GtUnpack
                     *process.horeco
                     *process.hfreco
                     *process.hbhereco
                     *process.zdcreco
                     *process.hcalMonitor
                     *process.hcalMonitorTasksOnlineSequence 
                     *process.hcalClient
                     *process.zdcMonitor
                     *process.zdcClient
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

