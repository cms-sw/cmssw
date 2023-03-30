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
from Configuration.Eras.Era_Run3_cff import Run3
process      = cms.Process('HCALDQM', Run3)
subsystem    = 'Hcal'
cmssw        = os.getenv("CMSSW_VERSION").split("_")
debugstr     = "### HcalDQM::cfg::DEBUG: "
warnstr      = "### HcalDQM::cfg::WARN: "
errorstr     = "### HcalDQM::cfg::ERROR:"
useOfflineGT = False
useFileInput = False
useMap       = False

unitTest = False
if 'unitTest=True' in sys.argv:
	unitTest=True
	useFileInput=False

#-------------------------------------
#	Central DQM Stuff imports
#-------------------------------------
from DQM.Integration.config.online_customizations_cfi import *
if useOfflineGT:
	process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
	process.GlobalTag.globaltag = autoCond['run3_data_prompt'] 
else:
	process.load('DQM.Integration.config.FrontierCondition_GT_cfi')
if unitTest:
	process.load("DQM.Integration.config.unittestinputsource_cfi")
	from DQM.Integration.config.unittestinputsource_cfi import options
elif useFileInput:
	process.load("DQM.Integration.config.fileinputsource_cfi")
	from DQM.Integration.config.fileinputsource_cfi import options
else:
	process.load('DQM.Integration.config.inputsource_cfi')
	from DQM.Integration.config.inputsource_cfi import options
process.load('DQM.Integration.config.environment_cfi')

#-------------------------------------
#	Central DQM Customization
#-------------------------------------
process.source.streamLabel = cms.untracked.string("streamDQMGPUvsCPU")
process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = 'HcalGPU'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'HcalGPU'
process.dqmSaverPB.runNumber = options.runNumber
process = customise(process)
process.DQMStore.verbose = 0
if not unitTest and not useFileInput :
  if not options.BeamSplashRun :
    process.source.minEventsPerLumi = 100

#-------------------------------------
#	CMSSW/Hcal non-DQM Related Module import
#-------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load('EventFilter.CastorRawToDigi.CastorRawToDigi_cff')
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


#-------------------------------------
#	Hcal DQM Tasks and Harvesters import
#	New Style
#-------------------------------------
process.load('DQM.HcalTasks.hcalGPUComparisonTask_cfi')
process.load('DQM.HcalTasks.HcalOnlineHarvesting')
process.load('DQM.HcalTasks.HcalQualityTests')

#-------------------------------------
#	Some Settings before Finishing up
#	New Style Modules
#-------------------------------------
oldsubsystem = subsystem
process.hcalGPUComparisonTask.tagHBHE_ref = "hltHbherecoLegacy"
process.hcalGPUComparisonTask.tagHBHE_target = "hltHbherecoFromGPU"
process.hcalGPUComparisonTask.runkeyVal = runType
process.hcalGPUComparisonTask.runkeyName = runTypeName

#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(
		process.hcalGPUComparisonTask
)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------

process.dqmPath = cms.EndPath(
		process.dqmEnv)
process.dqmPath1 = cms.EndPath(
		process.dqmSaver
		*process.dqmSaverPB
)

process.schedule = cms.Schedule(
	process.tasksPath,
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
process.options.wantSummary = True

# tracer
#process.Tracer = cms.Service("Tracer")
print("Final Source settings:", process.source)
process = customise(process)
