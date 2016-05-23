import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

#--------------------------------------------------
# Event Source and Condition

# Live Online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# Testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

# Required to load EcalMappingRecord
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#--------------------------------------------------
# DQM Environment

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = "L1T2016"
process.dqmSaver.tag = "L1T2016"
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference.root"

process.dqmEndPath = cms.EndPath(
    process.dqmEnv *
    process.dqmSaver
)

#--------------------------------------------------
# Standard Unpacking Path

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

process.rawToDigiPath = cms.Path(process.RawToDigi)

# For GCT, unpack all five samples.
process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)

process.gtDigis.DaqGtFedId = cms.untracked.int32(813)

#--------------------------------------------------
# Legacy DQM Paths

process.load("DQM.L1TMonitor.L1TMonitor_cff")
process.l1tMonitorEndPath = cms.EndPath(process.l1tMonitorEndPathSeq)

#--------------------------------------------------
# Stage2 DQM Paths

process.load("DQM.L1TMonitor.L1TStage2_cff")
process.l1tMonitorPath = cms.Path(process.l1tStage2online)

# Remove Subsystem Modules
#process.l1tStage2online.remove(process.l1tLayer1)
#process.l1tStage2online.remove(process.l1tStage2CaloLayer2)
#process.l1tStage2online.remove(process.l1tStage2uGMT)
#process.l1tStage2online.remove(process.l1tStage2uGt)
#process.l1tStage2online.remove(process.l1tStage2Bmtf)
#process.l1tStage2online.remove(process.l1tStage2Emtf)

#--------------------------------------------------
# Stage2 Unpacking Path

process.stage2UnpackPath = cms.Path(
    process.l1tCaloLayer1Digis +
    process.caloStage2Digis +
    process.gmtStage2Digis +
    process.gtStage2Digis +
    process.BMTFStage2Digis + 
    process.emtfStage2Digis
)

#--------------------------------------------------
# L1 Trigger DQM Schedule

process.schedule = cms.Schedule(
    process.rawToDigiPath,
    process.stage2UnpackPath,
    process.l1tMonitorPath,
    #process.l1tMonitorClientPath,
    process.l1tMonitorEndPath,
    #process.l1tMonitorClientEndPath,
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
process.bxTiming.FedSource = cms.untracked.InputTag("rawDataCollector")
process.l1s.fedRawData = cms.InputTag("rawDataCollector")
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
    process.bxTiming.FedSource = cms.untracked.InputTag("rawDataRepacker")
    process.l1s.fedRawData = cms.InputTag("rawDataRepacker")
    process.gtStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")

#--------------------------------------------------
# Process Customizations

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

