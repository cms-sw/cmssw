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
subsystem    = 'HcalCalib'
cmssw        = os.getenv("CMSSW_VERSION").split("_")
debugstr     = "### HcalDQM::cfg::DEBUG: "
warnstr      = "### HcalDQM::cfg::WARN: "
errorstr     = "### HcalDQM::cfg::ERROR:"
useOfflineGT = False
useFileInput = False
useMap       = False

#-------------------------------------
#	Central DQM Stuff imports
#-------------------------------------
from DQM.Integration.config.online_customizations_cfi import *
if useOfflineGT:
	process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
	process.GlobalTag.globaltag = '106X_dataRun3_HLT_Candidate_2019_11_26_14_48_16'
	#process.GlobalTag.globaltag = '100X_dataRun2_HLT_v1'
else:
	process.load('DQM.Integration.config.FrontierCondition_GT_cfi')
if useFileInput:
	process.load("DQM.Integration.config.fileinputsource_cfi")
	from DQM.Integration.config.fileinputsource_cfi import options
else:
	process.load('DQM.Integration.config.inputsource_cfi')
	from DQM.Integration.config.inputsource_cfi import options
process.load('DQM.Integration.config.environment_cfi')

#-------------------------------------
#	Central DQM Customization
#-------------------------------------
process.source.streamLabel = cms.untracked.string("streamDQMCalibration")
#process.source.SelectEvents = cms.untracked.vstring("*HcalCalibration*")
process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = subsystem
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = subsystem
process.dqmSaverPB.runNumber = options.runNumber
process = customise(process)
if not useFileInput:
	process.source.minEventsPerLumi=100


#-------------------------------------
#	CMSSW/Hcal non-DQM Related Module import
#-------------------------------------
#process.load('Configuration.Geometry.GeometryIdeal_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")
process.load("L1Trigger.Configuration.L1DummyConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

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
rawTag			= "hltHcalCalibrationRaw"
rawTagUntracked	= "hltHcalCalibrationRaw"
process.essourceSev = cms.ESSource(
		"EmptyESSource",
		recordName		= cms.string("HcalSeverityLevelComputerRcd"),
		firstValid		= cms.vuint32(1),
		iovIsRunNotTime	= cms.bool(True)
)
process.emulTPDigis = process.simHcalTriggerPrimitiveDigis.clone(
   inputLabel = ["hcalDigis", 'hcalDigis'],
   FrontEndFormatError = True,
   FG_threshold = 2,
   InputTagFEDRaw = rawTag
)

process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)
process.l1GtUnpack.DaqGtInputTag = rawTag
process.hbhereco = process.hbheprereco.clone()

#-------------------------------------
#	Hcal DQM Tasks and Clients import
#-------------------------------------
process.load("DQM.HcalTasks.PedestalTask")
process.load('DQM.HcalTasks.RawTask')
process.load("DQM.HcalTasks.LaserTask")
process.load("DQM.HcalTasks.LEDTask")
process.load("DQM.HcalTasks.UMNioTask")
process.load('DQM.HcalTasks.HcalOnlineHarvesting')
process.load("DQM.HcalTasks.HFRaddamTask")
process.load('DQM.HcalTasks.QIE11Task')

#-------------------------------------
#	To force using uTCA
#	Absent for Online Running
#-------------------------------------
if useMap:
    process.GlobalTag.toGet.append(
    	cms.PSet(record = cms.string("HcalElectronicsMapRcd"),
                 #tag = cms.string("HcalElectronicsMap_v7.05_hlt"),
                 tag = cms.string("HcalElectronicsMap_v9.0_hlt"),
        ))

#-------------------------------------
#	Some Settings before Finishing up
#-------------------------------------
process.hcalDigis.InputLabel = rawTag

process.hcalOnlineHarvesting.subsystem = subsystem
process.rawTask.subsystem = subsystem
process.rawTask.tagFEDs = rawTagUntracked
process.rawTask.tagReport = "hcalDigis"
process.rawTask.calibProcessing = True

#-------------------------------------
#	Prepare all the Laser Tasks
#-------------------------------------
process.hbhehpdTask = process.laserTask.clone(
   name = "HBHEHPDTask",
   laserType = 3,
   thresh_timingreflm_HO = [-1000., 1000.],
   thresh_timingreflm_HF = [-1000., 1000.],
   thresh_timingreflm_HB = [-1000., 1000.],
   thresh_timingreflm_HE = [-1000., 1000.]
)

process.hoTask = process.laserTask.clone(
   name = "HOTask",
   laserType = 4
)

process.hfTask = process.laserTask.clone(
   name = "HFTask",
   laserType = 5
)

process.hepmegaTask = process.laserTask.clone(
   name = "HEPMegaTask",
   laserType = 7
)

process.hemmegaTask = process.laserTask.clone(
  name = "HEMMegaTask",
  laserType = 8
)

process.hbpmegaTask = process.laserTask.clone(
  name = "HBPMegaTask",
  laserType = 9
)

process.hbmmegaTask = process.laserTask.clone(
  name = "HBMMegaTask",
  laserType = 10
)

process.qie11Task_laser = process.qie11Task.clone(
   name = "QIE11Task_laser",
   runkeyVal = runType,
   runkeyName = runTypeName,
   tagQIE11 = "hcalDigis",
   subsystem = "HcalCalib",
   laserType = 12
)

process.qie11Task_pedestal = process.qie11Task.clone(
   name = "QIE11Task_pedestal",
   runkeyVal = runType,
   runkeyName = runTypeName,
   tagQIE11 = "hcalDigis",
   subsystem = "HcalCalib",
   eventType = 1
)

process.ledTask.name = cms.untracked.string("LEDTask")

#-------------------------------------
#	Hcal DQM Tasks Sequence Definition
#-------------------------------------
process.tasksSequence = cms.Sequence(
		process.pedestalTask
		*process.hfRaddamTask
		*process.rawTask
		*process.hbhehpdTask
		*process.hoTask
		*process.hfTask
		*process.hepmegaTask
		*process.hemmegaTask
		*process.hbpmegaTask
		*process.hbmmegaTask
		*process.umnioTask
		*process.qie11Task_laser
		*process.qie11Task_pedestal
		*process.ledTask
)

process.harvestingSequence = cms.Sequence(
	process.hcalOnlineHarvesting
)

#-------------------------------------
#	Execution Sequence Definition
#-------------------------------------
process.p = cms.Path(
					process.hcalDigis
					*process.tasksSequence
					*process.harvestingSequence
                    *process.dqmEnv
                    *process.dqmSaver
		    *process.dqmSaverPB)

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
