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
from Configuration.StandardSequences.Eras import eras
process			= cms.Process('HCALDQM', eras.Run2_2018)
subsystem		= 'Hcal'
cmssw			= os.getenv("CMSSW_VERSION").split("_")
debugstr		= "### HcalDQM::cfg::DEBUG: "
warnstr			= "### HcalDQM::cfg::WARN: "
errorstr		= "### HcalDQM::cfg::ERROR:"
useOfflineGT	= True
useFileInput	= False
useMap		= False

# 2018 MWGR preparation: use text emap and a local run test file
MWGR1_hacks = True
useLocalFileInput = False

#-------------------------------------
#	Central DQM Stuff imports
#-------------------------------------
from DQM.Integration.config.online_customizations_cfi import *
if useFileInput:
	process.load("DQM.Integration.config.fileinputsource_cfi")
elif useLocalFileInput:
	process.source = cms.Source("HcalTBSource",
		fileNames = cms.untracked.vstring("file:/build/dryu/DQM/local_emulator/data/USC_308175.root"),
		maxEvents = cms.untracked.int32(1000),
		minEventsPerLumi = cms.untracked.int32(100),
	)
else:
	process.load('DQM.Integration.config.inputsource_cfi')
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
#process.source.minEventsPerLumi=100


#-------------------------------------
#	CMSSW/Hcal non-DQM Related Module import
#-------------------------------------
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load('EventFilter.CastorRawToDigi.CastorRawToDigi_cff')
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")
process.load('CondCore.CondDB.CondDB_cfi')
if useOfflineGT:
	#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
	process.GlobalTag.globaltag = '100X_dataRun2_HLT_v1' # '90X_dataRun2_HLT_v1'
else:
	process.load('DQM.Integration.config.FrontierCondition_GT_cfi')

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
if useLocalFileInput:
	isCosmicRun		= False
	isHeavyIon		= False
	runType = cms.untracked.int32(0)
	runTypeName = "pp"
else:
	runType			= process.runType.getRunType()
	runTypeName		= process.runType.getRunTypeName()
	isCosmicRun		= runTypeName=="cosmic_run" or runTypeName=="cosmic_run_stage1"
	isHeavyIon		= runTypeName=="hi_run"
cmssw			= os.getenv("CMSSW_VERSION").split("_")
if useLocalFileInput:
	rawTag = cms.InputTag("source")
	rawTagUntracked = cms.untracked.InputTag("source")
elif isHeavyIon:
	rawTag = cms.InputTag("rawDataRepacker")
	rawTagUntracked = cms.untracked.InputTag("rawDataRepacker")
	process.castorDigis.InputLabel = rawTag
else:
	rawTag			= cms.InputTag("rawDataCollector")
	rawTagUntracked = cms.untracked.InputTag("rawDataCollector")


process.emulTPDigis = \
		process.simHcalTriggerPrimitiveDigis.clone()
process.emulTPDigis.inputLabel = \
		cms.VInputTag("hcalDigis", 'hcalDigis')
process.emulTPDigis.FrontEndFormatError = \
		cms.bool(True)
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)
process.emulTPDigis.FG_threshold = cms.uint32(2)
process.emulTPDigis.InputTagFEDRaw = rawTag
process.emulTPDigis.upgradeHF = cms.bool(True)
process.emulTPDigis.upgradeHE = cms.bool(True)
process.emulTPDigis.inputLabel = cms.VInputTag("hcalDigis", "hcalDigis")
process.emulTPDigis.inputUpgradeLabel = cms.VInputTag("hcalDigis", "hcalDigis")
# Enable ZS on emulated TPs, to match what is done in data
process.emulTPDigis.RunZS = cms.bool(True)
process.emulTPDigis.ZS_threshold = cms.uint32(0)
process.hcalDigis.InputLabel = rawTag

# Exclude the laser FEDs. They contaminate the QIE10/11 digi collections. 
#from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
#run2_HCAL_2017.toModify(process.hcalDigis, FEDs=cms.untracked.vint32(724,725,726,727,728,729,730,731,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123))

#-------------------------------------
#	Hcal DQM Tasks and Harvesters import
#	New Style
#-------------------------------------
process.load("DQM.HcalTasks.DigiTask")
process.load('DQM.HcalTasks.TPTask')
process.load('DQM.HcalTasks.RawTask')
process.load('DQM.HcalTasks.NoCQTask')
#process.load('DQM.HcalTasks.ZDCTask')
process.load('DQM.HcalTasks.QIE11Task')
process.load('DQM.HcalTasks.HcalOnlineHarvesting')

#-------------------------------------
#	To force using uTCA
#	Will not be here for Online DQM
#-------------------------------------
if useMap:
	process.GlobalTag.toGet.append(cms.PSet(
		record = cms.string("HcalElectronicsMapRcd"),
		tag = cms.string("HcalElectronicsMap_v7.05_hlt"),
		)
	)
if MWGR1_hacks:
	#process.GlobalTag.toGet.append(cms.PSet(
	#	record = cms.string("PHcalRcd"),
	#	tag = cms.string("HCALRECO_Geometry_10YV4"),
	#	)
	#)
	#process.GlobalTag.toGet.append(cms.PSet(
	#	record = cms.string("HcalParametersRcd"),
	#	tag = cms.string("HCALParameters_Geometry_10YV4"),
	#	)
	#)
	process.es_pool = cms.ESSource("PoolDBESSource",
			DBParameters=process.CondDB,
			connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
			timetype = cms.string('runnumber'),
			toGet = cms.VPSet(
				cms.PSet(
					record = cms.string("HcalParametersRcd"),
					tag = cms.string("HCALParameters_Geometry_10YV4")
				),
				cms.PSet(
					record = cms.string("PHcalRcd"),
					tag = cms.string("HCALRECO_Geometry_10YV4")
				),
				cms.PSet(
					record = cms.string("HcalElectronicsMapRcd"),
					tag = cms.string("HcalElectronicsMap_all_2018")
				)
			),
			authenticationMethod = cms.untracked.uint32(0)
	)	
	process.es_prefer_es_pool = cms.ESPrefer('PoolDBESSource', 'es_pool')

	#process.es_ascii = cms.ESSource(
	#	'HcalTextCalibrations',
	#	input = cms.VPSet(
	#		cms.PSet(
	#			object = cms.string('ElectronicsMap'),
	#			#file = cms.FileInPath("HCALemap_all_J.txt")
	#			file = cms.FileInPath("emap_2018MWGR1.txt")
	#			#file = cms.FileInPath("emap_ngHF20170206_plus_HBHEP17_CRF.txt")
	#			#file = cms.FileInPath("emap_ngHF20170206_plus_HBHEP17_template.txt")
#	#			file = cms.FileInPath(settings.emapfileInPath)
	#			#file = cms.FileInPath('ngHF2017EMap_20170125_pre04.txt')
	#			#file = cms.FileInPath('ngHF2017EMap_20170206_pre05.txt')
	#		)
	#	)
	#)
	#process.es_prefer = cms.ESPrefer('HcalTextCalibrations', 'es_ascii')

#-------------------------------------
#	For Debugging
#-------------------------------------
#process.hcalDigiTask.moduleParameters.debug = 10

process.tbunpacker = cms.EDProducer(
	"HcalTBObjectUnpacker",
	IncludeUnmatchedHits	= cms.untracked.bool(False),
	HcalTriggerFED			= cms.untracked.int32(1)
)
process.tbunpacker.fedRawDataCollectionTag = rawTag

#-------------------------------------
#	Some Settings before Finishing up
#	New Style Modules
#-------------------------------------
oldsubsystem = subsystem
process.rawTask.tagFEDs = rawTagUntracked
process.digiTask.runkeyVal = runType
process.digiTask.runkeyName = runTypeName
process.nocqTask.runkeyVal = runType
process.nocqTask.runkeyName = runTypeName
process.rawTask.runkeyVal = runType
process.rawTask.runkeyName = runTypeName
process.tpTask.runkeyVal = runType
process.tpTask.runkeyName = runTypeName
#process.zdcTask.runkeyVal = runType
#process.zdcTask.runkeyName = runTypeName
#process.zdcTask.tagQIE10 = cms.untracked.InputTag("castorDigis")
process.qie11Task.runkeyVal = runType
process.qie11Task.runkeyName = runTypeName
process.qie11Task.tagQIE11 = cms.untracked.InputTag("hcalDigis")

#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(
		process.rawTask
		+process.digiTask
		#+process.tpTask
		+process.nocqTask
		+process.qie11Task
		#ZDC to be removed for 2017 pp running
		#+process.zdcTask
)

process.harvestingPath = cms.Path(
	process.hcalOnlineHarvesting
)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------
process.preRecoPath = cms.Path(
		process.tbunpacker
		*process.hcalDigis
		#*process.castorDigis
		#*process.emulTPDigis
)

process.dqmPath = cms.EndPath(
		process.dqmEnv)
process.dqmPath1 = cms.EndPath(
		process.dqmSaver
)

process.schedule = cms.Schedule(
	process.preRecoPath,
	process.tasksPath,
	#process.harvestingPath,
	process.dqmPath,
	process.dqmPath1
)

#-------------------------------------
#	Scheduling and Process Customizations
#-------------------------------------
process.options = cms.untracked.PSet(
		Rethrow = cms.untracked.vstring(
			"ProductNotFound",
			"TooManyProducts",
			"TooFewProducts"
		)
)
process.options.wantSummary = cms.untracked.bool(True)

# tracer
#process.Tracer = cms.Service("Tracer")
process = customise(process)
