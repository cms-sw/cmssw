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
subsystem		= 'HcalCalib'
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
	process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
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
process.source.streamLabel = cms.untracked.string("streamDQMCalibration")
process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = subsystem
referenceFileName = '/dqmdata/dqm/reference/hcal_reference.root'
process.DQMStore.referenceFileName = referenceFileName
process = customise(process)
process.source.SelectEvents = cms.untracked.vstring("*HcalCalibration*")

#-------------------------------------
#	CMSSW/Hcal non-DQM Related Module import
#-------------------------------------
process.load('Configuration.Geometry.GeometryIdeal_cff')
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("L1Trigger.Configuration.L1DummyConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")

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
cmssw			= os.getenv("CMSSW_VERSION").split("_")
rawTag			= cms.InputTag("hltHcalCalibrationRaw")
rawTagUntracked	= cms.InputTag("hltHcalCalibrationRaw")
process.l1GtUnpack.DaqGtInputTag = rawTag

#-------------------------------------
#	Hcal DQM Tasks and Clients import
#-------------------------------------
process.load("DQM.HcalTasks.LaserTask")
process.load("DQM.HcalTasks.PedestalTask")
process.load('DQM.HcalTasks.RadDamTask')

#-------------------------------------
#	To force using uTCA
#	Absent for Online Running
#-------------------------------------
if useMap:
    process.GlobalTag.toGet.append(cms.PSet(record = cms.string("HcalElectronicsMapRcd"),
                                            tag = cms.string("HcalElectronicsMap_v7.05_hlt"),
                                            )
                                   )

#-------------------------------------
#	For Debugginb
#-------------------------------------

#-------------------------------------
#	Some Settings before Finishing up
#-------------------------------------
process.hcalDigis.InputLabel = rawTag

#-------------------------------------
#	Hcal DQM Tasks Sequence Definition
#-------------------------------------
process.tasksSequence = cms.Sequence(
		process.laserTask
		*process.pedestalTask
		*process.raddamTask
)

#-------------------------------------
#	Execution Sequence Definition
#-------------------------------------
process.p = cms.Path(
					process.hcalDigis
					*process.tasksSequence
                    *process.dqmEnv
                    *process.dqmSaver)

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
