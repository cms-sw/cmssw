import FWCore.ParameterSet.Config as cms

process = cms.Process("L1TStage2EmulatorDQM")

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

#--------------------------------------------------
# DQM Environment

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = "L1T2016EMU"
process.dqmSaver.tag = "L1T2016EMU"
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1temu_reference.root"

process.dqmEndPath = cms.EndPath(
    process.dqmEnv *
    process.dqmSaver
)

#--------------------------------------------------
# Standard Unpacking Path

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

process.rawToDigiPath = cms.Path(process.RawToDigi)

# Remove Unpacker Modules
process.rawToDigiPath.remove(process.siStripDigis)
process.rawToDigiPath.remove(process.gtDigis)
process.rawToDigiPath.remove(process.gtEvmDigis)

#--------------------------------------------------
# Stage2 DQM Paths

# Filter fat events
from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
process.hltFatEventFilter = hltHighLevel.clone()
process.hltFatEventFilter.throw = cms.bool(False)
process.hltFatEventFilter.HLTPaths = cms.vstring('HLT_L1FatEvents_v*')

# This can be used if HLT filter not available in a run
process.selfFatEventFilter = cms.EDFilter("HLTL1NumberFilter",
        invert = cms.bool(False),
        period = cms.uint32(107),
        rawInput = cms.InputTag("rawDataCollector"),
        fedId = cms.int32(1024)
        )

process.load("DQM.L1TMonitor.L1TStage2Emulator_cff")

process.l1tEmulatorMonitorPath = cms.Path(
    process.hltFatEventFilter +
#    process.selfFatEventFilter +
    process.l1tStage2Unpack  +
    process.Stage2L1HardwareValidation +
    process.l1tStage2EmulatorOnlineDQM 
    )

# To get L1 CaloParams
# TODO: when L1 O2O is finished, this must be removed!
process.load('L1Trigger.L1TCalorimeter.caloStage2Params_cfi')
# To get CaloTPGTranscoder
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)

# To get L1 uGT parameters:
# TODO: when L1 O2O is finished, this must be removed!
#process.load('L1Trigger.L1TGlobal.hackConditions_cff')
process.load('L1Trigger.L1TGlobal.GlobalParameters_cff')

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
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print "Running with run type = ", process.runType.getRunType()
process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.csctfDigis.producer = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")
process.gtStage2Digis.InputLabel = cms.InputTag("rawDataCollector")

if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.gtStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")

#--------------------------------------------------
# Process Customizations

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
