#-------------------------------------
#	Hcal DQM Application using New DQM Sources/Clients
#	Online Mode
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
subsystem		= 'Hcal2'
cmssw			= os.getenv("CMSSW_VERSION").split("_")
debugstr		= "### HcalDQM::cfg::DEBUG: "
warnstr			= "### HcalDQM::cfg::WARN: "
errorstr		= "### HcalDQM::cfg::ERROR:"
useOfflineGT	= False
useFileInput	= False
useMap		= False
useMapText		= False

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
process.dqmSaver.tag = "Hcal2" # to have a file saved as DQM_V..._Hcal2...
referenceFileName = '/dqmdata/dqm/reference/hcal_reference.root'
process.DQMStore.referenceFileName = referenceFileName
process = customise(process)
process.DQMStore.verbose = 0

#	Note, runType is obtained after importing DQM-related modules
#	=> DQM-dependent
runType			= process.runType.getRunType()
print debugstr, "Running with run type= ", runType

#-------------------------------------
#	CMSSW/Hcal non-DQM Related Module import
#-------------------------------------
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
process.load('CondCore.CondDB.CondDB_cfi')

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
isCosmicRun     = runTypeName=="cosmic_run" or runTypeName=="cosmic_run_stage1"
isHeavyIon      = runTypeName=="hi_run"
cmssw			= os.getenv("CMSSW_VERSION").split("_")
rawTag			= cms.InputTag("rawDataCollector")
rawTagUntracked	= cms.untracked.InputTag("rawDataCollector")
if isHeavyIon:
	rawTag			= cms.InputTag("rawDataRepacker")
	rawTagUntracked	= cms.untracked.InputTag("rawDataRepacker")

process.hcalRecAlgos.DropChannelStatusBits = cms.vstring('')
#	init 2 emulators
process.emulTPPrim = \
		process.simHcalTriggerPrimitiveDigis.clone()
process.emulTPSec = \
		process.simHcalTriggerPrimitiveDigis.clone()
#	settings for emulators
process.emulTPPrim.inputLabel = \
		cms.VInputTag("primDigis", 'primDigis')
process.emulTPSec.inputLabel = \
		cms.VInputTag("secDigis", 'secDigis')
process.emulTPPrim.FrontEndFormatError = \
		cms.bool(True)
process.emulTPSec.FrontEndFormatError = \
		cms.bool(True)
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)
process.emulTPPrim.FG_threshold = cms.uint32(2)
process.emulTPPrim.InputTagFEDRaw = rawTag
process.emulTPSec.FG_threshold = cms.uint32(2)
process.emulTPSec.InputTagFEDRaw = rawTag
process.hbhereco = process.hbheprereco.clone()

#	UPDATES REQUESTED BY STEPH
process.hbheprereco.puCorrMethod = cms.int32(2) 
process.hbheprereco.ts4chi2 = cms.double(9999.) 
process.hbheprereco.timeMin = cms.double(-100.)
process.hbheprereco.timeMax = cms.double(100.)
process.hbheprereco.applyTimeConstraint = cms.bool(False) 

#	set the tag for default unpacker
process.hcalDigis.InputLabel = rawTag

#-------------------------------------
#	Hcal DQM Tasks and Clients import
#	New Style
#-------------------------------------
process.load('DQM.HcalTasks.RecHitTask')
process.load('DQM.HcalTasks.HcalOnlineHarvesting')
process.load('DQM.HcalTasks.DigiComparisonTask')
process.load('DQM.HcalTasks.TPComparisonTask')

#-------------------------------------
#	To force using uTCA
#	Will not be here for Online DQM
#-------------------------------------
if useMap:
	process.GlobalTag.toGet.append(cms.PSet(
		record = cms.string("HcalElectronicsMapRcd"),
		tag = cms.string("HcalElectronicsMap_v7.05_hlt")
		)
	)

#-------------------------------------
#	For Debugginb
#-------------------------------------

#-------------------------------------
#	prep 2 unpackers and assign the right label to the reco producer
#-------------------------------------
process.primDigis = process.hcalDigis.clone()
process.primDigis.InputLabel = rawTag
primFEDs = [x*2+1100 for x in range(9)]
primFEDs[len(primFEDs):] = [x+724 for x in range(8)]
primFEDs[len(primFEDs):] = [1118, 1120, 1122]
print "Primary FEDs to be Unpacked:", primFEDs
process.primDigis.FEDs = cms.untracked.vint32(primFEDs)
process.hbhereco.digiLabel = cms.InputTag("primDigis")
process.horeco.digiLabel = cms.InputTag("primDigis")
process.hfreco.digiLabel = cms.InputTag("primDigis")

process.secDigis = process.hcalDigis.clone()
process.secDigis.InputLabel = rawTag
process.secDigis.ElectronicsMap = cms.string("full")
secFEDs = [x+700 for x in range(18)]
print "Secondary FEDs to be Unpacked:", secFEDs
process.secDigis.FEDs = cms.untracked.vint32(secFEDs)

process.digiComparisonTask.tagHBHE1 = cms.untracked.InputTag('primDigis')
process.digiComparisonTask.tagHBHE2 = cms.untracked.InputTag('secDigis')
process.digiComparisonTask.runkeyVal = runType
process.digiComparisonTask.runkeyName = runTypeName
process.digiComparisonTask.subsystem = cms.untracked.string(subsystem)

process.tpComparisonTask.tag1 = cms.untracked.InputTag('primDigis')
process.tpComparisonTask.tag2 = cms.untracked.InputTag('secDigis')
process.tpComparisonTask.runkeyVal = runType
process.tpComparisonTask.runkeyName = runTypeName
process.tpComparisonTask.subsystem = cms.untracked.string(subsystem)

#-------------------------------------
#	Settigns for the Primary Modules
#-------------------------------------
oldsubsystem = subsystem
process.recHitTask.tagHBHE = cms.untracked.InputTag("hbhereco")
process.recHitTask.tagHO = cms.untracked.InputTag("horeco")
process.recHitTask.tagHF = cms.untracked.InputTag("hfreco")
process.recHitTask.runkeyVal = runType
process.recHitTask.runkeyName = runTypeName
process.recHitTask.tagRaw = rawTagUntracked
process.recHitTask.subsystem = cms.untracked.string(subsystem)

process.hcalOnlineHarvesting.subsystem = cms.untracked.string(subsystem)

#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(
		process.recHitTask
		+process.digiComparisonTask
		+process.tpComparisonTask
)

process.harvestingPath = cms.Path(
	process.hcalOnlineHarvesting
)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------
process.preRecoPath = cms.Path(
		process.primDigis
		*process.secDigis
		*process.emulTPPrim
		*process.emulTPSec
)

process.recoPath = cms.Path(
		process.hfreco
		*process.hbhereco
		*process.horeco
)

process.dqmPath = cms.Path(
		process.dqmEnv
		*process.dqmSaver
)

process.schedule = cms.Schedule(
		process.preRecoPath,
		process.recoPath,
		process.tasksPath,
		process.harvestingPath,
		process.dqmPath
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
