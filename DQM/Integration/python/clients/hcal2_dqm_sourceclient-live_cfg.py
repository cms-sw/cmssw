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

#
# these Modifiers are like eras as well, for more info check
# Configuration/StandardSequences/python/Eras.py
# PRocess accepts a (*list) of modifiers
#
from Configuration.StandardSequences.Eras import eras
process			= cms.Process('HCALDQM', 
    eras.run2_HCAL_2017, 
    eras.run2_HF_2017,
    eras.run2_HEPlan1_2017
)
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
	process.GlobalTag.globaltag = '90X_dataRun2_HLT_v1'
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
process.source.minEventsPerLumi=5

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
#process.load("RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi")
#process.load("RecoLocalCalo.HcalRecProducers.hbheplan1_cfi")
#process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi")
#process.load("RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi")
#from RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi import hfreco as _phase1_hfreco
#from RecoLocalCalo.HcalRecProducers.hbheplan1_cfi import hbheplan1
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

#	set the tag for default unpacker
process.hcalDigis.InputLabel = rawTag

#-------------------------------------
#	Hcal DQM Tasks and Clients import
#	New Style
#-------------------------------------
process.load('DQM.HcalTasks.RecHitTask')
process.load('DQM.HcalTasks.HcalOnlineHarvesting')
process.load('DQM.HcalTasks.DigiComparisonTask')

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
#	Settings for the Primary Modules
#-------------------------------------
oldsubsystem = subsystem
process.recHitTask.tagHBHE = cms.untracked.InputTag("hbheplan1")
process.recHitTask.tagHO = cms.untracked.InputTag("horeco")
process.recHitTask.tagHF = cms.untracked.InputTag("hfreco")
process.recHitTask.runkeyVal = runType
process.recHitTask.runkeyName = runTypeName
process.recHitTask.tagRaw = rawTagUntracked
process.recHitTask.subsystem = cms.untracked.string(subsystem)

process.hcalOnlineHarvesting.subsystem = cms.untracked.string(subsystem)

#-------------------------------------
#	Phase 1 upgrade modifiers
#-------------------------------------
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toModify(process.recHitTask, hfPreRecHitsAvailable=cms.untracked.bool(True))
run2_HF_2017.toModify(process.recHitTask, tagPreHF=cms.untracked.InputTag("hfprereco"))

#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(
		process.recHitTask
)

process.harvestingPath = cms.Path(
	process.hcalOnlineHarvesting
)

#-------------------------------------
process.digiPath = cms.Path(
		process.hcalDigis
)

process.recoPath = cms.Path(
    process.horeco
    *process.hfprereco
    *process.hfreco
    *process.hbheprereco
    *process.hbheplan1
)

process.dqmPath = cms.Path(
		process.dqmEnv
		*process.dqmSaver
)

process.schedule = cms.Schedule(
		process.digiPath,
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
