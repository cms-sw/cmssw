import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEMULATION')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Repacked_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')

# Select the Message Logger output you would like to see:
process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    #fileNames = cms.untracked.vstring("file:/d101/icali/ROOTFiles/HIHIghPtRAW/181530/A8D6061E-030D-E111-A482-BCAEC532971A.root")
	#fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V1-v1/00000/262AA156-744A-E311-9829-002618943945.root")
    #fileNames = cms.untracked.vstring("/store/RelVal/CMSSW_7_0_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_0_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("file:22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("file:test.root")
    fileNames = cms.untracked.vstring("file:/mnt/hadoop/cms/store/user/icali/HIHighPt/HIHIHighPt_RAW_Skim_HLT_HIFullTrack14/4d786c9deacb28bba8fe5ed87e99b9e4/SD_HIFullTrack14_975_1_SZU.root")
    )


process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('L1Emulator_HI_newLayer2.root'),
    dataset = cms.untracked.PSet(
    filterName = cms.untracked.string(''),
    dataTier = cms.untracked.string('')
    )
                                           )
process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')

process.RCTConverter = cms.EDProducer(
    "l1t::L1TCaloRCTToUpgradeConverter",
    regionTag = cms.InputTag("simRctDigis"),
    emTag = cms.InputTag("simRctDigis"))


process.caloStage1 = cms.EDProducer(
    "l1t::L1TCaloStage1Producer",
    CaloRegions = cms.InputTag("RCTConverter"),
    CaloEmCands = cms.InputTag("RCTConverter"),
    FirmwareVersion = cms.uint32(1)  ## 1=HI algo, 2= pp algo
    )

process.GCTConverter=cms.EDProducer("l1t::L1TCaloUpgradeToGCTConverter",
    InputCollection = cms.InputTag("caloStage1")
    )

process.load('L1Trigger.Configuration.SimL1Emulator_cff')
process.simRctDigis.ecalDigis = cms.VInputTag(cms.InputTag('ecalDigis:EcalTriggerPrimitives'))
process.simRctDigis.hcalDigis = cms.VInputTag(cms.InputTag('hcalDigis'))
process.simGtDigis.GctInputTag = 'GCTConverter'

process.digiStep = cms.Sequence(
    process.ecalDigis
    *process.hcalDigis
)

# process.simGctDigis = cms.Sequence(
#     process.RCTConverter
#     *process.caloStage1
#     *process.GCTConverter
# )

process.SimL1Emulator = cms.Sequence(
    process.simRctDigis +
    #process.simGctDigis +
    process.RCTConverter +
    process.caloStage1 +
    process.GCTConverter +
    process.SimL1MuTriggerPrimitives +
    process.SimL1MuTrackFinders +
    process.simRpcTriggerDigis +
    process.simGmtDigis +
    process.SimL1TechnicalTriggers +
    process.simGtDigis )

process.p1 = cms.Path(
    process.digiStep
    *process.SimL1Emulator
    )

process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.p1, process.output_step
    )

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
