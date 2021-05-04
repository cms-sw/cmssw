# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('L1')

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
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)


# Output definition

process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *",
					   "drop *_mix_*_*"),
    fileName = cms.untracked.string('L1T_EDM.root'),
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
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    threshold = cms.untracked.string('DEBUG')
)

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')


# user stuff

# raw data from MP card
process.load('EventFilter.L1TRawToDigi.stage1MP7BufferRaw_cfi')
process.stage1Raw.nFramesOffset    = cms.untracked.vuint32(0)
process.stage1Raw.nFramesLatency   = cms.untracked.vuint32(96)
process.stage1Raw.rxFile = cms.untracked.string("stage1_rx_summary.txt")
process.stage1Raw.txFile = cms.untracked.string("stage1_tx_summary.txt")

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("stage1Raw"),
    feds = cms.untracked.vint32 ( 1352 ),
    dumpPayload = cms.untracked.bool ( True )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.caloStage1Digis_cfi')
process.caloStage1Digis.InputLabel = cms.InputTag('stage1Raw')


process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.egToken = cms.InputTag("caloStage1Digis")
process.l1tStage2CaloAnalyzer.tauToken = cms.InputTag("caloStage1Digis:rlxTaus")
process.l1tStage2CaloAnalyzer.jetToken = cms.InputTag("caloStage1Digis")
process.l1tStage2CaloAnalyzer.etSumToken = cms.InputTag("caloStage1Digis")

# Path and EndPath definitions
process.path = cms.Path(
    process.stage1Raw
    +process.dumpRaw
    +process.caloStage1Digis
    +process.l1tStage2CaloAnalyzer
)

process.out = cms.EndPath(
    process.output
)
