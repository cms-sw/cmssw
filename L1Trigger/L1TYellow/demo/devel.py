import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEMULATION')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('L1Trigger/L1TYellow/upgrade_emulation_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

# Select the Firmware Version
#
process.load('L1Trigger/L1TYellow/l1tyellow_params_cfi')
from L1Trigger.L1TYellow.l1tyellow_params_cfi import yellowParams
yellowParams.firmwareVersion = cms.uint32(3)


# Select the Message Logger output you would like to see:
#
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('L1Trigger/L1TYellow/l1t_debug_messages_cfi')
#process.load('L1Trigger/L1TYellow/l1t_info_messages_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V1-v1/00000/262AA156-744A-E311-9829-002618943945.root")
    #fileNames = cms.untracked.vstring("/store/RelVal/CMSSW_7_0_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_0_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("file:test.root")
    )

process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('demo_output.root'),
    dataset = cms.untracked.PSet(
    filterName = cms.untracked.string(''),
    dataTier = cms.untracked.string('')
    ))


process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')

process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.debug = cms.EDAnalyzer("l1t::YellowParamsTester")

process.p1 = cms.Path(
    process.digiStep
    * process.debug
#    *process.dumpED
#    *process.dumpES
    )


process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.p1, process.output_step
    )

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
