#-------------------------------------
#	Hcal DQM Application using New DQM Sources/Clients
#	Old Modules are being run as well.
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
subsystem		= 'Hcal'
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
from DQM.Integration.config.online_customizations_cfi import *
if useOfflineGT:
	process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
	process.GlobalTag.globaltag = '74X_dataRun2_HLT_v1'
else:
	process.load('DQM.Integration.config.FrontierCondition_GT_cfi')
if useFileInput:
	process.load("DQM.Integration.config.fileinputsource_cfi")
else:
	process.load('DQM.Integration.config.inputsource_cfi')
process.load('DQMServices.Components.DQMEnvironment_cfi')
process.load('DQM.Integration.config.environment_cfi')

#-------------------------------------
#	Central DQM Customization
#-------------------------------------
process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = subsystem
referenceFileName = '/dqmdata/dqm/reference/hcal_reference.root'
process.DQMStore.referenceFileName = referenceFileName
process = customise(process)
process.DQMStore.verbose = 0

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
process.load("EventFilter.CastorRawToDigi.CastorRawToDigi_cff")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi")

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
#	Note, runType is obtained after importing DQM-related modules
#	=> DQM-dependent
#-------------------------------------
runType			= process.runType.getRunType()
runTypeName		= process.runType.getRunTypeName()
isCosmicRun		= runType==2 or runType==3
isHeavyIon		= runType==4
print debugstr, "Running with run type=%d name=%s notified=%s" % (
	runType, runTypeName, isCosmicRun)
cmssw			= os.getenv("CMSSW_VERSION").split("_")
rawTag			= cms.InputTag("rawDataCollector")
rawuntrackedTag = cms.untracked.InputTag("rawDataCollector")
if isHeavyIon:
	rawTag = cms.InputTag("rawDataRepacker")
	rawuntrackedTag = cms.untracked.InputTag("rawDataRepacker")
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
process.hcalDigis.InputLabel = rawTag
process.castorDigis.InputLabel = rawTag
process.zdcreco.digiLabel = cms.InputTag("castorDigis")

#-------------------------------------
#	Hcal DQM Tasks and Clients import
#	New Style
#-------------------------------------
process.load("DQM.HcalTasks.HcalDigiTask")
process.load("DQM.HcalTasks.HcalRawTask")
process.load("DQM.HcalTasks.HcalRecHitTask")
process.load("DQM.HcalTasks.HcalTPTask")
process.load("DQM.HcalTasks.HcalTimingTask")

#-------------------------------------
#	Hcal DQM Tasks and Clients Imports
#	Old Style.
#-------------------------------------
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
process.load("DQM.HcalMonitorTasks.HcalMonitorTasks_cfi")
process.load("DQM.HcalMonitorTasks.HcalTasksOnline_cff")
process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")
process.load("DQM.HcalMonitorModule.ZDCMonitorModule_cfi")
from DQM.HcalMonitorTasks.HcalMonitorTasks_cfi import SetTaskParams

#-------------------------------------
#	To force using uTCA
#	Will not be here for Online DQM
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
#process.hcalDigiTask.moduleParameters.debug = 10

#-------------------------------------
#	Some Settings before Finishing up
#	New Style Modules
#-------------------------------------
process.hcalDigiTask.moduleParameters.subsystem = cms.untracked.string(subsystem)
process.hcalRawTask.moduleParameters.subsystem = cms.untracked.string(subsystem)
process.hcalRecHitTask.moduleParameters.subsystem = cms.untracked.string(
		subsystem)
process.hcalTPTask.moduleParameters.subsystem = cms.untracked.string(subsystem)
process.hcalTimingTask.moduleParameters.subsystem = cms.untracked.string(
		subsystem)

#-------------------------------------
#	Some Settings before Finishing up
#	Old Style Modules
#-------------------------------------
process.hcalBeamMonitor.hotrate = 0.40

oldsubsystem = subsystem
if not oldsubsystem.endswith("/"):
	oldsubsystem+= "/"
process.hcalMonitor.subSystemFolder = oldsubsystem
SetTaskParams(process, "subSystemFolder", oldsubsystem)
process.hcalClient.subSystemFolder = oldsubsystem
process.hcalClient.baseHtmlDir = ''
process.hcalClient.databaseDir = '/nfshome0/hcaldqm/DQM_OUTPUT/ChannelStatus/'
process.hcalClient.databaseFirstUpdate = 10
process.hcalClient.databaseUpdateTime = 60
process.hcalClient.DeadCell_minerrorrate = 0.05
process.hcalClient.HotCell_mierrorrate = cms.untracked.double(0.10)
process.hcalClient.Beam_minerrorrate = cms.untracked.double(2.0)
process.hcalClient.isCosmicRun = cms.untracked.bool(isCosmicRun)
process.hcalDigiMonitor.maxDigiSizeHF = cms.untracked.int32(10)

process.hcalClient.enabledClients = [
	"DeadCellMonitor", "HotCellMonitor", "RecHitMonitor", "DigiMonitor",
	"RawDataMonitor", "TrigPrimMonitor", "NZSMonitor", "BeamMonitor",
	"CoarsePedestalMonitor", "DetDiagTimingMonitor", "Summary"
]
process.hcalDigis.ExpectedOrbitMessageTime = cms.untracked.int32(3559)
process.hcalDigiMonitor.ExpectedOrbitMessageTime = 3559
process.hcalDigiMonitor.shutOffOrbitTest = False
process.hcalDeadCellMonitor.excludeHORing2 = False
process.hcalCoarsePedestalMonitor.ADCDiffThresh = 2
process.hcalClient.CoarsePedestal_BadChannelStatusMask = cms.untracked.int32(
	(1<<5) | (1<<6))
process.hcalDataIntegrityMonitor.RawDataLabel = rawuntrackedTag
process.hcalDetDiagNoiseMonitor.RawDataLabel = rawuntrackedTag
process.hcalDetDiagTimingMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalNZSMonitor.RawDataLabel = rawuntrackedTag
process.hcalNoiseMonitor.RawDataLabel = rawuntrackedTag
process.hcalRawDataMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalDigiMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalHotCellMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalDeadCellMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalRecHitMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalBeamMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalTrigPrimMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalCoarsePedestalMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalDetDiagNoiseMonitor.FEDRawDataCollection = rawuntrackedTag
process.hcalNZSMonitor.FEDRawDataCollection = rawuntrackedTag
process.zdcMonitor.FEDRawDataCollection = rawuntrackedTag

#-------------------------------------
#	Some Settings before Finishing up
#	New Style Modules
#-------------------------------------
process.hcalDigiTask.moduleParameters.Labels.RAW = rawuntrackedTag
process.hcalRawTask.moduleParameters.Labels.RAW = rawuntrackedTag
process.hcalRecHitTask.moduleParameters.Labels.RAW = rawuntrackedTag
process.hcalTPTask.moduleParameters.Labels.RAW = rawuntrackedTag
process.hcalTimingTask.moduleParameters.Labels.RAW = rawuntrackedTag

#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksSequence = cms.Sequence(
		process.hcalDigiTask
		+process.hcalRawTask
		+process.hcalRecHitTask
		+process.hcalTPTask
		+process.hcalTimingTask
		+process.hcalMonitor
		+process.hcalMonitorTasksOnlineSequence
		+process.zdcMonitor
)

process.clientsSequence = cms.Sequence(
	process.hcalClient
)

#-------------------------------------
#	Quality Tester
#-------------------------------------
process.qTester = cms.EDAnalyzer(
	"QualityTester",
	prescaleFactor = cms.untracked.int32(1),
	qtList = cms.untracked.FileInPath(
		"DQM/HcalMonitorClient/data/hcal_qualitytest_config.xml"),
	getQualityTestsFromFile = cms.untracked.bool(True),
	qtestOnEndLumi = cms.untracked.bool(True),
	qtestOnEndRun = cms.untracked.bool(True)
)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------
process.preRecoSequence = cms.Sequence(
		process.hcalDigis
		*process.castorDigis
		*process.l1GtUnpack
)

process.recoSequence = cms.Sequence(
		process.emulTPDigis
		+process.hfreco
		+process.hbhereco
		+process.horeco
		+process.zdcreco
)

process.dqmSequence = cms.Sequence(
		process.dqmEnv
		*process.dqmSaver
)

process.p = cms.Path(
		process.preRecoSequence
		*process.recoSequence
		*process.tasksSequence
		*process.clientsSequence
		*process.qTester
		*process.dqmSequence
)

#process.schedule = cms.Schedule(process.p)

#-------------------------------------
#	Scheduling
#-------------------------------------
process.options = cms.untracked.PSet(
		Rethrow = cms.untracked.vstring(
			"ProductNotFound",
			"TooManyProducts",
			"TooFewProducts"
		)
)
