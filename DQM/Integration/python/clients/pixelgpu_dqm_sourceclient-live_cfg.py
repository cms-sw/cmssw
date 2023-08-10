#-------------------------------------
#	Pixel DQM Application using New DQM Sources/Clients
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
process      = cms.Process('PIXELDQMLIVE', Run3)
subsystem    = 'Pixel'
cmssw        = os.getenv("CMSSW_VERSION").split("_")
debugstr     = "### PixelDQM::cfg::DEBUG: "
warnstr      = "### PixelDQM::cfg::WARN: "
errorstr     = "### PixelDQM::cfg::ERROR:"
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
process.source.streamLabel = cms.untracked.string("streamDQMGPUvsCPU")
process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = 'PixelGPU'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'PixelGPU'
process.dqmSaverPB.runNumber = options.runNumber
process = customise(process)
process.DQMStore.verbose = 0
if not unitTest and not useFileInput :
  if not options.BeamSplashRun :
    process.source.minEventsPerLumi = 100

#-------------------------------------
#	CMSSW/Pixel non-DQM Related Module import
#-------------------------------------
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('FWCore.MessageLogger.MessageLogger_cfi')

#-------------------------------------
#	CMSSW non-DQM Related Module Settings
#-------------------------------------
runType			= process.runType.getRunType()
runTypeName		= process.runType.getRunTypeName()
isCosmicRun		= runTypeName=="cosmic_run" or runTypeName=="cosmic_run_stage1"
isHeavyIon		= runTypeName=="hi_run"
cmssw			= os.getenv("CMSSW_VERSION").split("_")

#-------------------------------------
#	Pixel DQM Tasks and Harvesters import
#-------------------------------------
process.load('DQM.SiPixelHeterogeneous.SiPixelHeterogenousDQM_FirstStep_cff')

#-------------------------------------
#	Some Settings before Finishing up
#-------------------------------------
process.siPixelPhase1RawDataErrorComparator.pixelErrorSrcGPU = cms.InputTag('hltSiPixelDigisFromSoA')
process.siPixelPhase1RawDataErrorComparator.pixelErrorSrcCPU = cms.InputTag('hltSiPixelDigisLegacy')

#-------------------------------------
#       Some Debug
#-------------------------------------
process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpPath = cms.Path(process.dump)

#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(process.siPixelPhase1RawDataErrorComparator)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------
process.dqmPath = cms.EndPath(process.dqmEnv)
process.dqmPath1 = cms.EndPath(process.dqmSaver*process.dqmSaverPB)
process.schedule = cms.Schedule(process.tasksPath,process.dumpPath,process.dqmPath,process.dqmPath1)

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
