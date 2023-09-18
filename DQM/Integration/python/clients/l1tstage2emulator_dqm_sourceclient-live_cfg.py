import FWCore.ParameterSet.Config as cms

import sys
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("L1TStage2EmulatorDQM", Run3)

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest=True

#--------------------------------------------------
# Event Source and Condition

if unitTest:
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
else:
    # Live Online DQM in P5
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options

# Testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

# Required to load Global Tag
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# required for EMTF emulator
process.load('Configuration.StandardSequences.MagneticField_cff')
# Required to load EcalMappingRecord
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#--------------------------------------------------
# DQM Environment

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = "L1TEMU"
process.dqmSaver.tag = "L1TEMU"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "L1TEMU"
process.dqmSaverPB.runNumber = options.runNumber

process.dqmEndPath = cms.EndPath(
    process.dqmEnv *
    process.dqmSaver *
    process.dqmSaverPB
)

#--------------------------------------------------
# Standard Unpacking Path

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

# remove unneeded unpackers
process.RawToDigi.remove(process.ecalDigis)
process.RawToDigi.remove(process.ecalPreshowerDigis)
process.RawToDigi.remove(process.hcalDigis)
process.RawToDigi.remove(process.muonDTDigis)
process.RawToDigi.remove(process.siPixelDigis)
process.RawToDigi.remove(process.siStripDigis)
process.RawToDigi.remove(process.castorDigis)
process.RawToDigi.remove(process.scalersRawToDigi)
process.RawToDigi.remove(process.tcdsDigis)
process.RawToDigi.remove(process.totemRPRawToDigi)
process.RawToDigi.remove(process.ctppsDiamondRawToDigi)
process.RawToDigi.remove(process.ctppsPixelDigis)

process.rawToDigiPath = cms.Path(process.RawToDigi)

#--------------------------------------------------
# Stage2 DQM Paths

# Filter fat events
from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
process.hltFatEventFilter = hltHighLevel.clone(
   throw = False,
# HLT_Physics now has the event % 107 filter as well as L1FatEvents
   HLTPaths = ['HLT_L1FatEvents_v*', 'HLT_Physics_v*']
)

# This can be used if HLT filter not available in a run
process.selfFatEventFilter = cms.EDFilter("HLTL1NumberFilter",
        invert = cms.bool(False),
        period = cms.uint32(107),
        rawInput = cms.InputTag("rawDataCollector"),
        fedId = cms.int32(1024)
        )

process.load("DQM.L1TMonitor.L1TStage2Emulator_cff")

process.l1tEmulatorMonitorPath = cms.Path(
    process.Stage2L1HardwareValidation +
    process.l1tStage2EmulatorOnlineDQM +
    process.hltFatEventFilter +
#    process.selfFatEventFilter +
    process.Stage2L1HardwareValidationForValidationEvents +
    process.l1tStage2EmulatorOnlineDQMValidationEvents
    )

# To get L1 conditions that are not in GlobalTag / O2O yet
#process.load("L1Trigger.L1TCalorimeter.hackConditions_cff")
#process.load("L1Trigger.L1TMuon.hackConditions_cff")
#process.gmtParams.caloInputsMasked = cms.bool(True) # Disable uGMT calo inputs like in the online configuration
#process.load("L1Trigger.L1TGlobal.hackConditions_cff")
process.load("L1Trigger.L1TGlobal.GlobalParameters_cff")

# To get CaloTPGTranscoder
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.HcalTPGCoderULUT.LUTGenerationMode = False

#--------------------------------------------------
# TODO: Stage2 Emulator Quality Tests
process.load("DQM.L1TMonitorClient.L1TStage2EmulatorMonitorClient_cff")
process.l1tStage2EmulatorMonitorClientPath = cms.Path(process.l1tStage2EmulatorMonitorClient)

#--------------------------------------------------
# Customize for other type of runs

# Cosmic run
if process.runType.getRunType() == process.runType.cosmic_run:
    pass

# Heavy-Ion run
if process.runType.getRunType() == process.runType.hi_run:
    process.hltFatEventFilter.HLTPaths.append('HLT_HIPhysics_v*')
    rawDataRepackerLabel = 'rawDataRepacker'
    process.onlineMetaDataDigis.onlineMetaDataInputLabel = rawDataRepackerLabel
    process.onlineMetaDataRawToDigi.onlineMetaDataInputLabel = rawDataRepackerLabel
    process.castorDigis.InputLabel = rawDataRepackerLabel
    process.ctppsDiamondRawToDigi.rawDataTag = rawDataRepackerLabel
    process.ctppsPixelDigis.inputLabel = rawDataRepackerLabel
    process.ecalDigis.cpu.InputLabel = rawDataRepackerLabel
    process.ecalPreshowerDigis.sourceTag = rawDataRepackerLabel
    process.hcalDigis.InputLabel = rawDataRepackerLabel
    process.muonCSCDigis.InputObjects = rawDataRepackerLabel
    process.muonDTDigis.inputLabel = rawDataRepackerLabel
    process.muonRPCDigis.InputLabel = rawDataRepackerLabel
    process.muonGEMDigis.InputLabel = rawDataRepackerLabel
    process.scalersRawToDigi.scalersInputTag = rawDataRepackerLabel
    process.siPixelDigis.cpu.InputLabel = rawDataRepackerLabel
    process.siStripDigis.ProductLabel = rawDataRepackerLabel
    process.tcdsDigis.InputLabel = rawDataRepackerLabel
    process.tcdsRawToDigi.InputLabel = rawDataRepackerLabel
    process.totemRPRawToDigi.rawDataTag = rawDataRepackerLabel
    process.totemTimingRawToDigi.rawDataTag = rawDataRepackerLabel
    process.csctfDigis.producer = rawDataRepackerLabel
    process.dttfDigis.DTTF_FED_Source = rawDataRepackerLabel
    process.gctDigis.inputLabel = rawDataRepackerLabel
    process.gtDigis.DaqGtInputTag = rawDataRepackerLabel
    process.twinMuxStage2Digis.DTTM7_FED_Source = rawDataRepackerLabel
    process.bmtfDigis.InputLabel = rawDataRepackerLabel
    process.valBmtfAlgoSel.feds = rawDataRepackerLabel
    process.omtfStage2Digis.inputLabel = rawDataRepackerLabel
    process.emtfStage2Digis.InputLabel = rawDataRepackerLabel
    process.gmtStage2Digis.InputLabel = rawDataRepackerLabel
    process.caloLayer1Digis.InputLabel = rawDataRepackerLabel
    process.caloStage1Digis.InputLabel = rawDataRepackerLabel
    process.caloStage2Digis.InputLabel = rawDataRepackerLabel
    process.simHcalTriggerPrimitiveDigis.InputTagFEDRaw = rawDataRepackerLabel
    process.l1tdeStage2CaloLayer1.fedRawDataLabel = rawDataRepackerLabel
    process.gtStage2Digis.InputLabel = rawDataRepackerLabel
    process.selfFatEventFilter.rawInput = rawDataRepackerLabel
    process.rpcTwinMuxRawToDigi.inputTag = rawDataRepackerLabel
    process.rpcCPPFRawToDigi.inputTag = rawDataRepackerLabel

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
print("Final Source settings:", process.source)
