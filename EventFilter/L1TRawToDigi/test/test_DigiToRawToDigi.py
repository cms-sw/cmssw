# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('DIGI2RAW2DIGI')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring("file:L1T_EDM.root")
)


# Output definition
process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *",
        "drop *_mix_*_*"),
    fileName = cms.untracked.string('L1T_PACK_EDM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

# enable debug message logging for our modules
process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations   = cms.untracked.vstring(
        'detailedInfo',
        'critical'
    ),
    detailedInfo   = cms.untracked.PSet(
        threshold  = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring(
        'l1tDigiToRaw', 'l1tRawToDigi'
    )
)


# plotting
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')

import L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi
process.packPlots = L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
process.packPlots.towerToken = cms.InputTag("l1tRawToDigi")
process.packPlots.clusterToken = cms.InputTag("None")
process.packPlots.egToken = cms.InputTag("None")
process.packPlots.tauToken = cms.InputTag("None")
process.packPlots.jetToken = cms.InputTag("l1tRawToDigi")
process.packPlots.etSumToken = cms.InputTag("l1tRawToDigi")

# plots from emulator
process.simPlots = L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
process.simPlots.towerToken = cms.InputTag("l1tDigis")
process.simPlots.clusterToken = cms.InputTag("None")
process.simPlots.egToken = cms.InputTag("None")
process.simPlots.tauToken = cms.InputTag("None")
process.simPlots.jetToken = cms.InputTag("l1tDigis")
process.simPlots.etSumToken = cms.InputTag("l1tDigis")

# user stuff
process.load("EventFilter.L1TRawToDigi.l1tDigiToRaw_cfi")
process.l1tDigiToRaw.InputLabel = cms.InputTag("l1tDigis")
process.load("EventFilter.L1TRawToDigi.l1tRawToDigi_cfi")

# Path and EndPath definitions
process.path = cms.Path(
    process.l1tDigiToRaw
    +process.l1tRawToDigi
    +process.simPlots
    +process.packPlots
)

process.out = cms.EndPath(
    process.output
)
