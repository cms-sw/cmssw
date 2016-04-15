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
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

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
runTypeName		= process.runType.getRunTypeName()
isCosmicRun		= runTypeName=="cosmic_run" or runTypeName=="cosmic_run_stage1"
isHeavyIon		= runTypeName=="hi_run"
cmssw			= os.getenv("CMSSW_VERSION").split("_")
rawTag			= cms.InputTag("rawDataCollector")
rawTagUntracked = cms.untracked.InputTag("rawDataCollector")
if isHeavyIon:
	rawTag = cms.InputTag("rawDataRepacker")
	rawTagUntracked = cms.untracked.InputTag("rawDataRepacker")

process.emulTPDigis = \
		process.simHcalTriggerPrimitiveDigis.clone()
process.emulTPDigis.inputLabel = \
		cms.VInputTag("hcalDigis", 'hcalDigis')
process.emulTPDigis.FrontEndFormatError = \
		cms.bool(True)
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)
process.emulTPDigis.FG_threshold = cms.uint32(2)
process.emulTPDigis.InputTagFEDRaw = rawTag
process.hcalDigis.InputLabel = rawTag

#-------------------------------------
#	Hcal DQM Tasks and Harvesters import
#	New Style
#-------------------------------------
process.load("DQM.HcalTasks.DigiTask")
process.load('DQM.HcalTasks.TPTask')
process.load('DQM.HcalTasks.RawTask')
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

#-------------------------------------
#	For Debugginb
#-------------------------------------
#process.hcalDigiTask.moduleParameters.debug = 10

#-------------------------------------
#	Some Settings before Finishing up
#	New Style Modules
#-------------------------------------
oldsubsystem = subsystem
process.rawTask.tagFEDs = rawTagUntracked
process.digiTask.runkeyVal = runType
process.digiTask.runkeyName = runTypeName
process.rawTask.runkeyVal = runType
process.rawTask.runkeyName = runTypeName
process.tpTask.runkeyVal = runType
process.tpTask.runkeyName = runTypeName

#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(
		process.rawTask
		+process.digiTask
		+process.tpTask
)

process.harvestingPath = cms.Path(
	process.hcalOnlineHarvesting
)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------
process.preRecoPath = cms.Path(
		process.hcalDigis
		*process.emulTPDigis
)

process.dqmPath = cms.EndPath(
		process.dqmEnv)
process.dqmPath1 = cms.EndPath(
		process.dqmSaver
)

process.schedule = cms.Schedule(
	process.preRecoPath,
	process.tasksPath,
	process.harvestingPath,
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
