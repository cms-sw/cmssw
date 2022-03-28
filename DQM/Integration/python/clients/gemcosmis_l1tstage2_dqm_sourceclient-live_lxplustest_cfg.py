import FWCore.ParameterSet.Config as cms

import sys
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
#process = cms.Process("L1TStage2DQM", Run2_2018)
#process.load('Configuration.StandardSequences.MagneticField_cff')

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("L1TStage2DQM", Run3)
process.load('Configuration.StandardSequences.MagneticField_cff')

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
    print ("***")
#    process.load("DQM.Integration.config.inputsource_cfi")
#    from DQM.Integration.config.inputsource_cfi import options

# # Testing in lxplus
process.load("DQM.Integration.config.fileinputsource_lxplustest_cfi")
from DQM.Integration.config.fileinputsource_lxplustest_cfi import options
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.debugModules = ['L1T']
process.MessageLogger.cout = cms.untracked.PSet(
#     threshold=cms.untracked.string('DEBUG'),
#    threshold = cms.untracked.string('INFO'),
    threshold = cms.untracked.string('WARNING'),
#    threshold = cms.untracked.string('ERROR'),
    DEBUG=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    INFO=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    WARNING=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    ERROR=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    default = cms.untracked.PSet( 
        limit=cms.untracked.int32(-1)  
    )
)


# Required to load Global Tag
# use different GT to work with 2018
process.load("DQM.Integration.config.FrontierCondition_GT_cfi") 
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '110X_mcRun3_2021_realistic_v6', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_mc_FULL', '')

# # Condition for lxplus: change and possibly customise the GT
# from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
# process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

# Required to load EcalMappingRecord
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#--------------------------------------------------
# DQM Environment

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = "L1T"
process.dqmSaver.tag = "L1T"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "L1T"
process.dqmSaverPB.runNumber = options.runNumber

process.dqmEndPath = cms.EndPath(process.dqmEnv * process.dqmSaver * process.dqmSaverPB)

#--------------------------------------------------
# Standard Unpacking Path

process.load("Configuration.StandardSequences.RawToDigi_cff")    #no difference from RawToDigi_Data_cff

# remove unneeded unpackers
process.RawToDigi.remove(process.ecalPreshowerDigis)
#process.RawToDigi.remove(process.muonCSCDigis)
#process.RawToDigi.remove(process.muonDTDigis)
#process.RawToDigi.remove(process.muonRPCDigis)
process.RawToDigi.remove(process.siPixelDigis)
process.RawToDigi.remove(process.siStripDigis)
process.RawToDigi.remove(process.castorDigis)
process.RawToDigi.remove(process.scalersRawToDigi)
process.RawToDigi.remove(process.tcdsDigis)
process.RawToDigi.remove(process.totemTriggerRawToDigi)
process.RawToDigi.remove(process.totemRPRawToDigi)
process.RawToDigi.remove(process.ctppsDiamondRawToDigi)
process.RawToDigi.remove(process.ctppsPixelDigis)

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
    process.l1tStage2OnlineDQM 
#    process.hltFatEventFilter +  #comment locally to run faster Nov 04 2020
#    process.selfFatEventFilter +
#    process.l1tStage2OnlineDQMValidationEvents
)

# Remove DQM Modules
## test running emulator for MC Nov 05 2020, remove all dqm module except EMTF
process.l1tStage2OnlineDQM.remove(process.l1tStage2BmtfSecond)
process.l1tStage2OnlineDQM.remove(process.l1tStage2BmtfZeroSupp)
process.l1tStage2OnlineDQM.remove(process.l1tStage2Omtf)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTIntermediateBMTF)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTIntermediateOMTFNeg)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTIntermediateOMTFPos)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTIntermediateEMTFNeg)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTIntermediateEMTFPos)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTZeroSupp)
process.l1tStage2OnlineDQM.remove(process.l1tStage2BmtfOutVsuGMTIn)
process.l1tStage2OnlineDQM.remove(process.l1tStage2OmtfOutVsuGMTIn)
process.l1tStage2OnlineDQM.remove(process.l1tStage2EmtfOutVsuGMTIn)
process.l1tStage2OnlineDQM.remove(process.l1tObjectsTiming)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTCaloLayer2Comp)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTOutVsuGTIn)
process.l1tStage2OnlineDQM.remove(process.l1tStage2BmtfZeroSuppFatEvts)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTZeroSuppFatEvts)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTMuonVsuGMTMuonCopy1)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTMuonVsuGMTMuonCopy2)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTMuonVsuGMTMuonCopy3)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTMuonVsuGMTMuonCopy4)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMTMuonVsuGMTMuonCopy5)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTMuon1vsMuon2)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTMuon1vsMuon3)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTMuon1vsMuon4)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTMuon1vsMuon5)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTMuon1vsMuon6)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTCalo1vsCalo2)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTCalo1vsCalo3)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTCalo1vsCalo4)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTCalo1vsCalo5)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGTCalo1vsCalo6)
# remove all except EMTF

process.l1tStage2OnlineDQM.remove(process.l1tStage2CaloLayer1)
process.l1tStage2OnlineDQM.remove(process.l1tStage2CaloLayer2)
process.l1tStage2OnlineDQM.remove(process.l1tStage2Bmtf)
#process.l1tStage2OnlineDQM.remove(process.l1tStage2Emtf)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGMT)
process.l1tStage2OnlineDQM.remove(process.l1tStage2uGT)

#--------------------------------------------------
# Stage2 Quality Tests
process.load("DQM.L1TMonitorClient.L1TStage2MonitorClient_cff")
process.l1tStage2MonitorClientPath = cms.Path(process.l1tStage2MonitorClient)

#--------------------------------------------------
# Customize for other type of runs

# Cosmic run
if (process.runType.getRunType() == process.runType.cosmic_run):
    # Remove Quality Tests for L1T Muon Subsystems since they are not optimized yet for cosmics
    process.l1tStage2MonitorClient.remove(process.l1TStage2uGMTQualityTests)
    process.l1tStage2MonitorClient.remove(process.l1TStage2EMTFQualityTests)
    #process.l1tStage2MonitorClient.remove(process.l1TStage2OMTFQualityTests)
    process.l1tStage2MonitorClient.remove(process.l1TStage2BMTFQualityTests)
    process.l1tStage2MonitorClient.remove(process.l1TStage2MuonQualityTestsCollisions)
    process.l1tStage2EventInfoClient.DisableL1Systems = cms.vstring("EMTF", "OMTF", "BMTF", "uGMT")

# Heavy-Ion run
if (process.runType.getRunType() == process.runType.hi_run):
    process.onlineMetaDataDigis.onlineMetaDataInputLabel = cms.InputTag("rawDataRepacker")
    process.onlineMetaDataRawToDigi.onlineMetaDataInputLabel = cms.InputTag("rawDataRepacker")
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataRepacker")
    process.ctppsPixelDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonGEMDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.tcdsDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.tcdsRawToDigi.InputLabel = cms.InputTag("rawDataRepacker")
    process.totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataRepacker")
    process.totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataRepacker")
    process.totemTimingRawToDigi.rawDataTag = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.twinMuxStage2Digis.DTTM7_FED_Source = cms.InputTag("rawDataRepacker")
    process.bmtfDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.omtfStage2Digis.inputLabel = cms.InputTag("rawDataRepacker")
    process.emtfStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.gmtStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.caloLayer1Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.caloStage1Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.caloStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.gtStage2Digis.InputLabel = cms.InputTag("rawDataRepacker")
    process.l1tStage2CaloLayer1.fedRawDataLabel = cms.InputTag("rawDataRepacker")
    process.l1tStage2uGMTZeroSupp.rawData = cms.InputTag("rawDataRepacker")
    process.l1tStage2uGMTZeroSuppFatEvts.rawData = cms.InputTag("rawDataRepacker")
    process.l1tStage2BmtfZeroSupp.rawData = cms.InputTag("rawDataRepacker")
    process.l1tStage2BmtfZeroSuppFatEvts.rawData = cms.InputTag("rawDataRepacker")
    process.selfFatEventFilter.rawInput = cms.InputTag("rawDataRepacker")
    process.rpcTwinMuxRawToDigi.inputTag = cms.InputTag("rawDataRepacker")
    process.rpcCPPFRawToDigi.inputTag = cms.InputTag("rawDataRepacker")

#--------------------------------------------------
# L1T Online DQM Schedule

#test using emulator for MC Nov 05 2020
# process.options = cms.untracked.PSet(
#     FailPath = cms.untracked.vstring(),
#     IgnoreCompletely = cms.untracked.vstring(),
#     Rethrow = cms.untracked.vstring(),
#     SkipEvent = cms.untracked.vstring(),
#     allowUnscheduled = cms.obsolete.untracked.bool,
#     canDeleteEarly = cms.untracked.vstring(),
#     deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
#     dumpOptions = cms.untracked.bool(False),
#     emptyRunLumiMode = cms.obsolete.untracked.string,
#     eventSetup = cms.untracked.PSet(
#         forceNumberOfConcurrentIOVs = cms.untracked.PSet(
#             allowAnyLabel_=cms.required.untracked.uint32
#         ),
#         numberOfConcurrentIOVs = cms.untracked.uint32(0)
#     ),
#     fileMode = cms.untracked.string('FULLMERGE'),
#     forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
#     makeTriggerResults = cms.obsolete.untracked.bool,
#     numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
#     numberOfConcurrentRuns = cms.untracked.uint32(1),
#     numberOfStreams = cms.untracked.uint32(0),
#     numberOfThreads = cms.untracked.uint32(8),
#     printDependencies = cms.untracked.bool(False),
#     sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
#     throwIfIllegalParameter = cms.untracked.bool(True),
#     wantSummary = cms.untracked.bool(False)
# )


#process.simMuonGEMDigis.InputLabel = cms.InputTag("rawDataRepacker")

process.load('EventFilter.L1TRawToDigi.emtfStage2Digis_cfi')
# process.TFileService = cms.Service("TFileService", fileName = cms.string("DQM_GEM_348683.root") )

# process.endjob_step = cms.EndPath(process.endOfProcess)

process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring('keep *',),
    # outputCommands = cms.untracked.vstring('drop *',
    #                                        'keep *_*_*_L1TEMULATION'),
    fileName = cms.untracked.string('stage1_debug.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
        )
    )

process.output_step = cms.EndPath(process.output)

###process.muonGEMDigis.useDBEMap = False

RawToDigi_AWB = cms.Sequence(
    process.muonRPCDigis +
    process.muonCSCDigis +
    process.muonGEMDigis

)

process.raw2digis_step = cms.Path(RawToDigi_AWB)


emtfRawToDigi = cms.Sequence(
   process.emtfStage2Digis
)

process.emtfraw2digi = cms.Path(emtfRawToDigi)

process.schedule = cms.Schedule(
    process.emtfraw2digi,
#    process.rawToDigiPath,
#    process.raw2digis_step,

    process.l1tMonitorPath,
#    process.l1tStage2MonitorClientPath, #comment locally to run faster Nov 04 2020
   # process.l1tMonitorEndPath
   # process.output_step
    process.dqmEndPath
)

#--------------------------------------------------
# Process Customizations

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

