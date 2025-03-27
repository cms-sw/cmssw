#-------------------------------------
#	PF DQM Application using New DQM Sources/Clients 
#   Taken from HCAL DQM Client
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
process      = cms.Process('PFDQM', Run3)
subsystem    = 'ParticleFlow'
cmssw        = os.getenv("CMSSW_VERSION").split("_")
debugstr     = "### PFDQM::cfg::DEBUG: "
warnstr      = "### PFDQM::cfg::WARN: "
errorstr     = "### PFDQM::cfg::ERROR:"
useOfflineGT = False
useFileInput = False
useMap       = False
unitTest     = 'unitTest=True' in sys.argv

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
    process.load("DQM.Integration.config.unitteststreamerinputsource_cfi")
    from DQM.Integration.config.unitteststreamerinputsource_cfi import options
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

if not useFileInput:
    # stream label
    if process.runType.getRunType() == process.runType.hi_run:
        process.source.streamLabel = "streamHIDQMGPUvsCPU"
    else:
        process.source.streamLabel = "streamDQMGPUvsCPU"

process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = 'PFGPU'
process.dqmSaver.runNumber = options.runNumber
# process.dqmSaverPB.tag = 'PFGPU'
# process.dqmSaverPB.runNumber = options.runNumber
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

#-------------------------------------
#	CMSSW/Hcal non-DQM Related Module Settings
#	-> runType
#	-> Generic Input tag for the Raw Collection
#	-> cmssw version
#----------------------------------------------

runType			= process.runType.getRunType()
runTypeName		= process.runType.getRunTypeName()
isCosmicRun		= runTypeName=="cosmic_run" or runTypeName=="cosmic_run_stage1"
isHeavyIon		= runTypeName=="hi_run"
cmssw			= os.getenv("CMSSW_VERSION").split("_")


#-------------------------------------
#	PF DQM Tasks and Harvesters import
#	New Style
#-------------------------------------
process.load('DQM.PFTasks.pfHcalGPUComparisonTask_cfi')

#-------------------------------------
#	Some Settings before Finishing up
#	New Style Modules
#-------------------------------------
oldsubsystem = subsystem
#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(
        process.pfHcalGPUComparisonTask
)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------

process.dqmPath = cms.EndPath(
		process.dqmEnv)
process.dqmPath1 = cms.EndPath(
		process.dqmSaver
		#*process.dqmSaverPB
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
			"TooManyProducts",
			"TooFewProducts"
		),
        TryToContinue = cms.untracked.vstring('ProductNotFound')
)
process.options.wantSummary = True

# tracer
#process.Tracer = cms.Service("Tracer")
process = customise(process)
print("Global Tag used:", process.GlobalTag.globaltag.value())
print("Final Source settings:", process.source)
