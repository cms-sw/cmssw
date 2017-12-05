import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process("L1TStage2DQM", eras.Run2_2017)

#--------------------------------------------------
# Event Source and Condition

# Live Online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")

# Testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

# Required to load Global Tag
process.load("DQM.Integration.config.FrontierCondition_GT_cfi") 

# Required to load EcalMappingRecord
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#--------------------------------------------------
# DQM Environment

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = "L1T"
process.dqmSaver.tag = "L1T"
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference.root"

process.dqmEndPath = cms.EndPath(process.dqmEnv * process.dqmSaver)

#--------------------------------------------------
# Standard Unpacking Path

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

process.rawToDigiPath = cms.Path(process.RawToDigi)

#--------------------------------------------------
# Stage2 Unpacker and DQM Path

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

process.load("DQM.L1TMonitor.L1TStage2_cff")

process.l1tMonitorPath = cms.Path(
    process.l1tStage2Unpack +
    process.l1tStage2OnlineDQM +
    process.hltFatEventFilter +
#    process.selfFatEventFilter +
    process.l1tStage2OnlineDQMValidationEvents
)

# Remove DQM Modules
#process.l1tStage2online.remove(process.l1tStage2CaloLayer1)
#process.l1tStage2online.remove(process.l1tStage2CaloLayer2)
#process.l1tStage2online.remove(process.l1tStage2Bmtf)
#process.l1tStage2online.remove(process.l1tStage2Emtf)
#process.l1tStage2online.remove(process.l1tStage2uGMT)
#process.l1tStage2online.remove(process.l1tStage2uGt)

#--------------------------------------------------
# Stage2 Quality Tests
process.load("DQM.L1TMonitorClient.L1TStage2MonitorClient_cff")
process.l1tStage2MonitorClientPath = cms.Path(process.l1tStage2MonitorClient)

#--------------------------------------------------
# Legacy DQM EndPath
# TODO: Is lumi scalers still relevant?

#process.load("DQM.L1TMonitor.L1TMonitor_cff")
#process.l1tMonitorEndPath = cms.EndPath(process.l1tMonitorEndPathSeq)

#--------------------------------------------------
# Customize for other type of runs

# Cosmic run
if (process.runType.getRunType() == process.runType.cosmic_run):
    process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference_cosmic.root"
    # Remove Quality Tests for L1T Muon Subsystems since they are not optimized yet for cosmics
    process.l1tStage2MonitorClient.remove(process.l1TStage2uGMTQualityTests)
    process.l1tStage2MonitorClient.remove(process.l1TStage2EMTFQualityTests)
    process.l1tStage2MonitorClient.remove(process.l1TStage2BMTFQualityTests)
    process.l1tStage2EventInfoClient.DisableL1Systems = cms.vstring("EMTF", "OMTF", "BMTF", "uGMT")

# Heavy-Ion run
if (process.runType.getRunType() == process.runType.hi_run):
    process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference_hi.root"
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataRepacker")
    process.ctppsPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.tcdsDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.tcdsRawToDigi.InputLabel = cms.InputTag("rawDataRepacker")
    process.totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataRepacker")
    process.totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.twinMuxStage2Digis.DTTM7_FED_Source = cms.InputTag("rawDataRepacker")
    process.bmtfDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.emtfStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.gmtStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.l1tCaloLayer1Digis.fedRawDataLabel = cms.InputTag("rawDataRepacker")
    process.caloStage1Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.caloStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.gtStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.l1tStage2CaloLayer1.fedRawDataLabel = cms.InputTag("rawDataRepacker")
    process.l1tStage2uGMTZeroSupp.rawData = cms.InputTag("rawDataRepacker")
    process.l1tStage2uGMTZeroSuppFatEvts.rawData = cms.InputTag("rawDataRepacker")
    process.selfFatEventFilter.rawInput = cms.InputTag("rawDataRepacker")

#--------------------------------------------------
# L1T Online DQM Schedule

process.schedule = cms.Schedule(
    process.rawToDigiPath,
    process.l1tMonitorPath,
    process.l1tStage2MonitorClientPath,
#    process.l1tMonitorEndPath,
    process.dqmEndPath
)

#--------------------------------------------------
# Process Customizations

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

