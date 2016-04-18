import FWCore.ParameterSet.Config as cms

process = cms.Process("L1TStage2EmulatorDQM")

#--------------------------------------------------
# Event Source and Condition

# Live Online DQM in P5
#process.load("DQM.Integration.config.inputsource_cfi")
#process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# Testing in lxplus
process.load("DQM.Integration.config.fileinputsource_cfi")
process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

# Required to load EcalMappingRecord
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#--------------------------------------------------
# DQM Environment

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = "L1TEMU2016"
process.dqmSaver.tag = "L1TEMU2016"
#process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1temu_reference.root"

process.dqmEndPath = cms.EndPath(process.dqmEnv * process.dqmSaver)

#--------------------------------------------------
# Standard Unpacking Path

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

process.rawToDigiPath = cms.Path(process.RawToDigi)

# Remove Unpacker Modules
process.rawToDigiPath.remove(process.siStripDigis)
process.rawToDigiPath.remove(process.gtDigis)
process.rawToDigiPath.remove(process.gtEvmDigis)

#--------------------------------------------------
# Stage2 Unpacker, Emulator, and Emulator DQM Path

process.load("DQM.L1TMonitor.L1TStage2_cff")
process.load("DQM.L1TMonitor.L1TStage2Emulator_cff")

process.l1tEmulatorMonitorPath = cms.Path(
    process.l1tStage2Emulator *
    process.l1tStage2Unpack +
    process.l1tStage2EmulatorOnlineDQM
)

#--------------------------------------------------
# L1T Emulator Online DQM Schedule

process.dumpEventContent = cms.EDAnalyzer(
    "EventContentAnalyzer",
    verbose = cms.untracked.bool(True),
    verboseForModuleLabels = cms.untracked.vstring("")
)

process.schedule = cms.Schedule( 
    process.rawToDigiPath,
    process.l1tEmulatorMonitorPath,
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

#--------------------------------------------------
# Process Customizations

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

