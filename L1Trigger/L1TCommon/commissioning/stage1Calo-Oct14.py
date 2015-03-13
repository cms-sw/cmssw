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
    input = cms.untracked.int32(9)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)


# Output definition
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *"),
    fileName = cms.untracked.string('L1T_EDM.root')
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
        'mp7BufferDumpToRaw',
        'l1tDigis',
	'caloStage1Digis'
    )
)

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')


# user stuff

# raw data from MP card
import EventFilter.L1TRawToDigi.mp7BufferDumpToRaw_cfi
process.stage1Raw = EventFilter.L1TRawToDigi.mp7BufferDumpToRaw_cfi.mp7BufferDumpToRaw.clone()
process.stage1Raw.fedId           = cms.untracked.int32(1)
process.stage1Raw.packetisedData  = cms.untracked.bool(False)
process.stage1Raw.rxFile          = cms.untracked.string("rx_summary.txt")
process.stage1Raw.txFile          = cms.untracked.string("tx_summary.txt")
process.stage1Raw.nFramesPerEvent = cms.untracked.int32(6)
process.stage1Raw.txLatency       = cms.untracked.int32(98)
process.stage1Raw.nRxLinks        = cms.untracked.int32(38)
process.stage1Raw.nTxLinks        = cms.untracked.int32(38)
process.stage1Raw.nRxEventHeaders = cms.untracked.int32(0)
process.stage1Raw.nTxEventHeaders = cms.untracked.int32(0)
process.stage1Raw.rxBlockLength   = cms.untracked.vint32(
    6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,
    0,0)

process.stage1Raw.txBlockLength   = cms.untracked.vint32(
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    6,6)

# raw to digi
#import EventFilter.L1TRawToDigi.l1tRawToDigi_cfi
#process.l1tDigis = EventFilter.L1TRawToDigi.l1tRawToDigi_cfi.l1tRawToDigi.clone()
#process.l1tDigis.FedId = cms.int32(2)
#process.l1tDigis.InputLabel = cms.InputTag("rawData")

#process.l1tDigis.Unpackers = cms.vstring([ "l1t::CaloTowerUnpackerFactory",
#                                           "l1t::EGammaUnpackerFactory",
#                                           "l1t::EtSumUnpackerFactory",
#                                           "l1t::JetUnpackerFactory",
#                                           "l1t::TauUnpackerFactory",
#                                           "l1t::MPUnpackerFactory"])

### emulator ###



### diagnostics ###

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("stage1Raw"),
    feds = cms.untracked.vint32 ( 1 ),
    dumpPayload = cms.untracked.bool ( True )
)

# plots from unpacker
#import L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi
#process.rawPlots = L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
#process.rawPlots.towerToken = cms.InputTag("l1tDigis")
#process.rawPlots.clusterToken = cms.InputTag("None")
#process.rawPlots.egToken = cms.InputTag("None")
#process.rawPlots.tauToken = cms.InputTag("None")
#process.rawPlots.jetToken = cms.InputTag("l1tDigis")
#process.rawPlots.etSumToken = cms.InputTag("l1tDigis")

# plots from emulator
#process.simPlots = L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
#process.simPlots.towerToken = cms.InputTag("caloStage2Digis")
#process.simPlots.clusterToken = cms.InputTag("caloStage2Digis")
#process.simPlots.egToken = cms.InputTag("caloStage2Digis")
#process.simPlots.tauToken = cms.InputTag("caloStage2Digis")
#process.simPlots.jetToken = cms.InputTag("caloStage2Digis")
#process.simPlots.etSumToken = cms.InputTag("caloStage2Digis")


# Path and EndPath definitions
process.path = cms.Path(

    # produce RAW
    process.stage1Raw
    +process.dumpRaw

)

process.out = cms.EndPath(
    process.output
)
