import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process("L1TStage2EmulatorDQM", eras.Run2_2016)

#--------------------------------------------------
# Event Source and Condition

# Live Online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Due to the GT override in the above include, we have trouble with
# conflicting CaloParams from stage1 and stage2.  This workaround
# can go away once either the es_prefer is removed from DQM or the
# new L1TCaloStage2ParamsRcd is integrated into CMSSW.
if 'es_prefer_GlobalTag' in process.__dict__:
    process.__dict__.pop('es_prefer_GlobalTag')
    process._Process__esprefers.pop('es_prefer_GlobalTag')

# Testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

# Required to load EcalMappingRecord
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

# Required for EMTF emulation
process.load('Configuration.StandardSequences.MagneticField_cff')

#--------------------------------------------------
# DQM Environment

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = "L1TEMU"
process.dqmSaver.tag = "L1TEMU"
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1temu_reference.root"

process.dqmEndPath = cms.EndPath(
    process.dqmEnv *
    process.dqmSaver
)

#--------------------------------------------------
# Standard Unpacking Path

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

process.rawToDigiPath = cms.Path(process.RawToDigi)

#--------------------------------------------------
# Stage2 DQM Paths

# Filter fat events
from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
process.hltFatEventFilter = hltHighLevel.clone()
process.hltFatEventFilter.throw = cms.bool(False)
# HLT_Physics now has the event % 107 filter as well as L1FatEvents
process.hltFatEventFilter.HLTPaths = cms.vstring('HLT_L1FatEvents_v*', 'HLT_Physics_v*')

# This can be used if HLT filter not available in a run
process.selfFatEventFilter = cms.EDFilter("HLTL1NumberFilter",
        invert = cms.bool(False),
        period = cms.uint32(107),
        rawInput = cms.InputTag("rawDataCollector"),
        fedId = cms.int32(1024)
        )

process.load("DQM.L1TMonitor.L1TStage2Emulator_cff")

process.l1tEmulatorMonitorPath = cms.Path(
    process.l1tStage2Unpack  +
    process.Stage2L1HardwareValidation +
    process.l1tStage2EmulatorOnlineDQM +
    process.hltFatEventFilter +
#    process.selfFatEventFilter +
    process.l1tStage2UnpackValidationEvents  +
    process.Stage2L1HardwareValidationForValidationEvents +
    process.l1tStage2EmulatorOnlineDQMValidationEvents
    )

# To get L1 conditions that are not in GlobalTag / O2O yet
process.load("L1Trigger.L1TCalorimeter.hackConditions_cff")
process.load("L1Trigger.L1TMuon.hackConditions_cff")
process.gmtParams.caloInputsMasked = cms.bool(True) # Disable uGMT calo inputs like in the online configuration
process.load("L1Trigger.L1TGlobal.hackConditions_cff")

# To get CaloTPGTranscoder
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)

#--------------------------------------------------
# TODO: Stage2 Emulator Quality Tests
process.load("DQM.L1TMonitorClient.L1TStage2EmulatorMonitorClient_cff")
process.l1tStage2EmulatorMonitorClientPath = cms.Path(process.l1tStage2EmulatorMonitorClient)

#--------------------------------------------------
# L1T Emulator Online DQM Schedule

process.schedule = cms.Schedule( 
    process.rawToDigiPath,
    process.l1tEmulatorMonitorPath,
    process.l1tStage2EmulatorMonitorClientPath,
    process.dqmEndPath
)

#--------------------------------------------------
# Process Customizations

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
