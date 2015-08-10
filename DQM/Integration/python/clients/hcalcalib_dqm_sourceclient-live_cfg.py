#-------------------------------------
#	Hcal DQM Application using New DQM Sources/Clients
#-------------------------------------

#-------------------------------------
#	Standard Python Imports
#-------------------------------------
import os, sys, socket, string

#-------------------------------------
#	Standard CMSSW Imports/Definitions
#-------------------------------------
import FWCore.ParameterSet.Config as cms
process			= cms.Process('HCALDQM')
subsystem		= 'HcalCalib'
cmssw			= os.getenv("CMSSW_VERSION").split("_")
debugstr		= "### HcalDQM::cfg::DEBUG: "
warnstr			= "### HcalDQM::cfg::WARN: "
errorstr		= "### HcalDQM::cfg::ERROR:"
useOfflineGT	= False
useFileInput	= False
useMap		= False

#-------------------------------------
#	Central DQM Stuff imports
#-------------------------------------
from DQM.Integration.test.online_customizations_cfi import *
if useOfflineGT:
	process.load('DQM.Integration.test.FrontierCondition_GT_Offline_cfi')
	process.GlobalTag.globaltag = '74X_dataRun2_Prompt_v1'
else:
	process.load('DQM.Integration.test.FrontierCondition_GT_cfi')
if useFileInput:
	process.load("DQM.Integration.test.fileinputsource_cfi")
else:
	process.load('DQM.Integration.test.inputsource_cfi')
process.load('DQMServices.Components.DQMEnvironment_cfi')
process.load('DQM.Integration.test.environment_cfi')

#-------------------------------------
#	Central DQM Customization
#-------------------------------------
process.source.streamLabel = cms.untracked.string("streamDQMCalibration")
process.dqmEnv.subSystemFolder = subsystem
referenceFileName = '/dqmdata/dqm/reference/hcal_reference.root'
process.DQMStore.referenceFileName = referenceFileName
process = customise(process)
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

#	Note, runType is obtained after importing DQM-related modules
#	=> DQM-dependent
runType			= process.runType.getRunType()
print debugstr, "Running with run type= ", runType

#-------------------------------------
#	CMSSW/Hcal non-DQM Related Module import
#-------------------------------------
process.load('Configuration.Geometry.GeometryIdeal_cff')
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("L1Trigger.Configuration.L1DummyConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

#-------------------------------------
#	CMSSW/Hcal non-DQM Related Module Settings
#	-> runType
#	-> Generic Input tag for the Raw Collection
#	-> cmssw version
#	-> Turn off default blocking of dead channels from rechit collection
#	-> Drop Channel Status Bits (had benn 'HcalCellOff', "HcalCellDead")
#	-> For Trigger Primitives Emulation
#	-> L1 GT setting
#	-> Rename the hbheprereco to hbhereco
#-------------------------------------
runType			= process.runType.getRunType()
cmssw			= os.getenv("CMSSW_VERSION").split("_")
rawTag			= cms.InputTag("hltHcalCalibrationRaw")
process.essourceSev = cms.ESSource(
		"EmptyESSource",
		recordName		= cms.string("HcalSeverityLevelComputerRcd"),
		firstValid		= cms.vuint32(1),
		iovIsRunNotTime	= cms.bool(True)
)
process.hcalRecAlgos.DropChannelStatusBits = cms.vstring('')
process.emulTPDigis = \
		process.simHcalTriggerPrimitiveDigis.clone()
process.emulTPDigis.inputLabel = \
		cms.VInputTag("hcalDigis", 'hcalDigis')
process.emulTPDigis.FrontEndFormatError = \
		cms.bool(True)
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)
process.emulTPDigis.FG_threshold = cms.uint32(2)
process.emulTPDigis.InputTagFEDRaw = rawTag
process.l1GtUnpack.DaqGtInputTag = rawTag
process.hbhereco = process.hbheprereco.clone()

#-------------------------------------
#	Hcal DQM Tasks and Clients import
#-------------------------------------
process.load("DQM.HcalTasks.HcalLEDTask")
process.load("DQM.HcalTasks.HcalLaserTask")
process.load("DQM.HcalTasks.HcalPedestalTask")

#-------------------------------------
#	To force using uTCA
#	Absent for Online Running
#-------------------------------------
if useMap:
	process.es_pool = cms.ESSource("PoolDBESSource",
			process.CondDBSetup,
			timetype = cms.string('runnumber'),
			toGet = cms.VPSet(
				cms.PSet(
					record = cms.string(
						"HcalElectronicsMapRcd"
					),
					tag = cms.string(
						"HcalElectronicsMap_v7.05_hlt"
					)
				)
			),
			connect = cms.string(
				'frontier://FrontierProd/CMS_CONDITIONS'),
			authenticationMethod = cms.untracked.uint32(0)
	)	
	process.es_prefer_es_pool = cms.ESPrefer('PoolDBESSource', 'es_pool')

#-------------------------------------
#	For Debugginb
#-------------------------------------
#process.hcalTPTask.moduleParameters.debug = 0

#-------------------------------------
#	Some Settings before Finishing up
#-------------------------------------
process.hcalDigis.InputLabel = rawTag
process.hcalLEDTask.moduleParameters.subsystem = cms.untracked.string(subsystem)
process.hcalLEDTask.moduleParameters.calibTypes = cms.untracked.vint32(
		1,2,3,4,5)
process.hcalLEDTask.moduleParameters.Labels.RAW = cms.untracked.InputTag(
		"hltHcalCalibrationRaw")
process.hcalLaserTask.moduleParameters.subsystem = cms.untracked.string(subsystem)
process.hcalLaserTask.moduleParameters.calibTypes = cms.untracked.vint32(
		1,2,3,4,5)
process.hcalLaserTask.moduleParameters.Labels.RAW = cms.untracked.InputTag(
		"hltHcalCalibrationRaw")
process.hcalPedestalTask.moduleParameters.subsystem = cms.untracked.string(
		subsystem)
process.hcalPedestalTask.moduleParameters.calibTypes = cms.untracked.vint32(
		1,2,3,4,5)
process.hcalPedestalTask.moduleParameters.Labels.RAW = cms.untracked.InputTag(
		"hltHcalCalibrationRaw")

#-------------------------------------
#	Hcal DQM Tasks Sequence Definition
#-------------------------------------
process.tasksSequence = cms.Sequence(
		process.hcalLEDTask
		*process.hcalLaserTask
		*process.hcalPedestalTask
)

<<<<<<< HEAD:DQM/Integration/python/test/hcalcalib_dqm_sourceclient-live_cfg.py
#-------------------------------------
#	Execution Sequence Definition
#-------------------------------------
process.p = cms.Path(
					process.hcalDigis
					*process.tasksSequence
                    *process.dqmEnv
                    *process.dqmSaver)

#-------------------------------------
#	Scheduling
#-------------------------------------
process.options = cms.untracked.PSet(
		Rethrow = cms.untracked.vstring(
#			"ProductNotFound",
			"TooManyProducts",
			"TooFewProducts"
		),
		SkipEvent = cms.untracked.vstring(
			'ProductNotFound'
		)
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
