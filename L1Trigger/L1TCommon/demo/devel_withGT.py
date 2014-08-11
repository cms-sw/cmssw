import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEMULATION')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')

# Select the Message Logger output you would like to see:

process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('L1Trigger/L1TYellow/l1t_debug_messages_cfi')
#process.load('L1Trigger/L1TYellow/l1t_info_messages_cfi')

process.load('L1Trigger/L1TCalorimeter/caloStage1Params_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    #fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V1-v1/00000/262AA156-744A-E311-9829-002618943945.root")
    #fileNames = cms.untracked.vstring("/store/RelVal/CMSSW_7_0_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_0_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/22610530-FC24-E311-AF35-003048FFD7C2.root")
    fileNames = cms.untracked.vstring("file:22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("file:test.root")
    )


process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    #outputCommands = cms.untracked.vstring('keep *'),
    outputCommands = cms.untracked.vstring('drop *',
                                           'keep *_*_*_L1TEMULATION'),
    fileName = cms.untracked.string('demo_output_full.root'),
    dataset = cms.untracked.PSet(
    filterName = cms.untracked.string(''),
    dataTier = cms.untracked.string('')
    )
                                           )
process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')

process.rctLayer2Format = cms.EDProducer(
    "l1t::L1TCaloRCTToUpgradeConverter",
    regionTag = cms.InputTag("simRctDigis"),
    emTag = cms.InputTag("simRctDigis"))

process.Layer2HW = cms.EDProducer(
    "l1t::Stage1Layer2Producer",
    CaloRegions = cms.InputTag("rctLayer2Format"),
    CaloEmCands = cms.InputTag("rctLayer2Format"),
    FirmwareVersion = cms.uint32(1)  ## 1=HI algo, 2= pp algo
    )

process.Layer2Phys = cms.EDProducer("l1t::PhysicalEtAdder",
                                      InputCollection = cms.InputTag("Layer2HW")
)

process.Layer2gctFormat = cms.EDProducer("l1t::L1TCaloUpgradeToGCTConverter",
    InputCollection = cms.InputTag("Layer2Phys")
    )


process.load('L1Trigger.Configuration.SimL1Emulator_cff')
process.simGtDigis.GctInputTag = 'Layer2gctFormat'

# overwrite old simGctDigis
process.simGctDigis = cms.Sequence(
        process.rctLayer2Format
        *process.Layer2HW
        *process.Layer2Phys
        *process.Layer2gctFormat
        )


process.p1 = cms.Path(
#    process.digiStep
    #process.Stage1GCT
    process.SimL1Emulator
    )



process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.p1, process.output_step
    )

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
