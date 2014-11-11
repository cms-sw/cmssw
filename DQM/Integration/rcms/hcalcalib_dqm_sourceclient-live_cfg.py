import FWCore.ParameterSet.Config as cms
from DQM.HcalMonitorTasks.HcalMonitorTasks_cfi import SetTaskParams

import os, sys, socket
# Get Host information
host = socket.gethostname().split('.')[0].lower()
HcalPlaybackHost='dqm-c2d07-13'.lower()
HcalCalibPlaybackHost='dqm-c2d07-16'.lower()
#HcalPlaybackHost='srv-c2d04-25'.lower()
#HcalCalibPlaybackHost='srv-c2d04-28'.lower()


playbackHCALCALIB=False
if (host==HcalCalibPlaybackHost):
    playbackHCALCALIB=True
    

process = cms.Process("HCALDQM")

subsystem="HcalCalib"

#----------------------------
# Event Source
#-----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'Hcal Orbit Gap DQM Consumer'
process.EventStreamHttpReader.SelectEvents =  cms.untracked.PSet(SelectEvents = cms.vstring('HLT_HcalCalibration'))
process.EventStreamHttpReader.sourceURL = cms.string('http://%s:23100/urn:xdaq-application:lid=30' % socket.gethostname())


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
# Hcal DQM Source, including SimpleReconstrctor
#-----------------------------
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi")

# Corrections to HF timing
process.hfreco.firstSample = 3
process.hfreco.samplesToAdd = 4

# ZDC Corrections to reco
process.zdcreco.firstSample  = 4
process.zdcreco.samplesToAdd = 3
process.zdcreco.recoMethod   = 2

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
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)

process.p = cms.Path(process.hcalDigis
                     *process.l1GtUnpack
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

