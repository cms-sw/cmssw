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
process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = subsystem
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = subsystem
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
rawTag			= "rawDataCollector"
rawTagUntracked = "rawDataCollector"
if isHeavyIon:
	rawTag = "rawDataRepacker"
	rawTagUntracked = "rawDataRepacker"
	process.castorDigis.InputLabel = rawTag

process.emulTPDigis = process.simHcalTriggerPrimitiveDigis.clone(
   inputLabel = ["hcalDigis", 'hcalDigis'],
   FrontEndFormatError = True,
   FG_threshold = 2,
   InputTagFEDRaw = rawTag,
   upgradeHF = True,
   upgradeHE = True,
   upgradeHB = True,
   inputUpgradeLabel = ["hcalDigis", "hcalDigis"],
   # Enable ZS on emulated TPs, to match what is done in data
   RunZS = True,
   ZS_threshold = 0
)

process.hcalDigis.InputLabel = rawTag
process.emulTPDigisNoTDCCut = process.emulTPDigis.clone(
     parameters = cms.untracked.PSet(
	ADCThresholdHF = cms.uint32(255),
	TDCMaskHF = cms.uint64(0xFFFFFFFFFFFFFFFF)
     )
)
process.HcalTPGCoderULUT.LUTGenerationMode = False

# For sent-received comparison
process.load("L1Trigger.Configuration.L1TRawToDigi_cff")
# For heavy ion runs, need to reconfigure sources for L1TRawToDigi
if isHeavyIon:
	process.csctfDigis.producer = "rawDataRepacker"
	process.dttfDigis.DTTF_FED_Source = "rawDataRepacker"
	process.twinMuxStage2Digis.DTTM7_FED_Source = "rawDataRepacker"
	process.omtfStage2Digis.inputLabel = "rawDataRepacker"
	process.caloStage1Digis.InputLabel = "rawDataRepacker" #new
	process.bmtfDigis.InputLabel = "rawDataRepacker"
	process.emtfStage2Digis.InputLabel = "rawDataRepacker"
	process.caloLayer1Digis.InputLabel = "rawDataRepacker" #not sure
	process.caloStage2Digis.InputLabel = "rawDataRepacker"
	process.gmtStage2Digis.InputLabel = "rawDataRepacker"
	process.gtStage2Digis.InputLabel = "rawDataRepacker"
	process.rpcTwinMuxRawToDigi.inputTag = "rawDataRepacker"
	process.rpcCPPFRawToDigi.inputTag = "rawDataRepacker"

# Exclude the laser FEDs. They contaminate the QIE10/11 digi collections. 
#from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
#run2_HCAL_2017.toModify(process.hcalDigis, FEDs=[724,725,726,727,728,729,730,731,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123])

#-------------------------------------
#	Hcal DQM Tasks and Harvesters import
#	New Style
#-------------------------------------
process.load("DQM.HcalTasks.DigiTask")
process.load('DQM.HcalTasks.TPTask')
process.load('DQM.HcalTasks.RawTask')
process.load('DQM.HcalTasks.NoCQTask')
process.load('DQM.HcalTasks.FCDTask')
process.load('DQM.HcalTasks.ZDCTask')
#process.load('DQM.HcalTasks.QIE11Task') # 2018: integrate QIE11Task into DigiTask
process.load('DQM.HcalTasks.HcalOnlineHarvesting')
process.load('DQM.HcalTasks.HcalQualityTests')
process.load('DQM.HcalTasks.hcalMLTask_cfi')

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
process.nocqTask.runkeyVal = runType
process.nocqTask.runkeyName = runTypeName
process.rawTask.runkeyVal = runType
process.rawTask.runkeyName = runTypeName
process.tpTask.runkeyVal = runType
process.tpTask.runkeyName = runTypeName
#process.zdcTask.runkeyVal = runType
#process.zdcTask.runkeyName = runTypeName
#process.zdcTask.tagQIE10 = cms.untracked.InputTag("castorDigis")
#process.qie11Task.runkeyVal = runType
#process.qie11Task.runkeyName = runTypeName
#process.qie11Task.tagQIE11 = cms.untracked.InputTag("hcalDigis")
process.fcdTask.runkeyVal = runType
process.fcdTask.runkeyName = runTypeName

#-------------------------------------
#	Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(
		process.rawTask
		+process.digiTask
		+process.tpTask
		+process.nocqTask
		+process.fcdTask
		#+process.qie11Task
		#ZDC to be removed after 2018 PbPb run
		+process.zdcQIE10Task
		+process.hcalMLTask
)

if isHeavyIon:
    process.tasksPath += process.zdcQIE10Task

process.harvestingPath = cms.Path(
	process.hcalOnlineHarvesting
)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------
process.preRecoPath = cms.Path(
		process.hcalDigis
		#*process.castorDigis # not in Run3
		*process.emulTPDigis
		*process.emulTPDigisNoTDCCut
		*process.L1TRawToDigi
)

process.dqmPath = cms.EndPath(
		process.dqmEnv)
process.dqmPath1 = cms.EndPath(
		process.dqmSaver
		*process.dqmSaverPB
)
process.qtPath = cms.Path(process.hcalQualityTests)

process.schedule = cms.Schedule(
	process.preRecoPath,
	process.tasksPath,
	process.qtPath,
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
process.options.wantSummary = True

# tracer
#process.Tracer = cms.Service("Tracer")
print("Final Source settings:", process.source)
process = customise(process)
