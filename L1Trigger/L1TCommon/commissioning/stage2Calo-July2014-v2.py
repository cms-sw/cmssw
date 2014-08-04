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
	'caloStage2TowerDigis',
	'caloStage2Digis'
    )
)

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')


# user stuff

# raw data from MP card
import EventFilter.L1TRawToDigi.mp7BufferDumpToRaw_cfi
process.stage2Layer2Raw = EventFilter.L1TRawToDigi.mp7BufferDumpToRaw_cfi.mp7BufferDumpToRaw.clone()
process.stage2Layer2Raw.fedId           = cms.untracked.int32(2)
process.stage2Layer2Raw.rxFile          = cms.untracked.string("rx_summary.txt")
process.stage2Layer2Raw.txFile          = cms.untracked.string("tx_summary.txt")
process.stage2Layer2Raw.nFramesPerEvent = cms.untracked.int32(54)
process.stage2Layer2Raw.txLatency       = cms.untracked.int32(54)
process.stage2Layer2Raw.nRxEventHeaders = cms.untracked.int32(1)
process.stage2Layer2Raw.nTxEventHeaders = cms.untracked.int32(0)
process.stage2Layer2Raw.rxBlockLength   = cms.untracked.vint32(
    40,0,40,0,40,0,40,0,40,
    0,40,0,40,0,40,0,40,0,
    40,0,40,0,40,0,40,0,40,
    0,40,0,40,0,40,0,40,0,
    40,0,40,0,40,0,40,0,40,
    0,40,0,40,0,40,0,40,0,
    40,0,40,0,40,0,40,0,40,
    0,40,0,40,0,40,0,40,0)

process.stage2Layer2Raw.txBlockLength   = cms.untracked.vint32(
    39,39,39,39,39,39,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0)

# raw data from Demux
#process.stage2DemuxRaw = EventFilter.L1TRawToDigi.mp7BufferDumpToRaw_cfi.mp7BufferDumpToRaw.clone()
#process.stage2DemuxRaw.fedId = cms.untracked.int32(1)
#process.stage2DemuxRaw.rxFile = cms.untracked.string("")
#process.stage2DemuxRaw.txFile = cms.untracked.string("tx_summary.txt")
#process.stage2DemuxRaw.txLatency = cms.untracked.int32(54)
#process.stage2DemuxRaw.nRxEventHeaders = cms.untracked.int32(0)
#process.stage2DemuxRaw.nTxEventHeaders = cms.untracked.int32(0)
#process.stage2DemuxRaw.rxBlockLength    = cms.untracked.vint32(
#    39,39,39,39,39,39,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0)
# demux output as seen in data
#process.stage2DemuxRaw.txBlockLength    = cms.untracked.vint32(
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,6,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,4,0,0,0,0,0)
# demux output as specified in docs
#process.stage2DemuxRaw.txBlockLength    = cms.untracked.vint32(
#    12,4,12,8,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0)


# merge raw data
import EventFilter.RawDataCollector.rawDataCollector_cfi
process.rawData = EventFilter.RawDataCollector.rawDataCollector_cfi.rawDataCollector.clone()
process.rawData.RawCollectionList = cms.VInputTag(
    cms.InputTag('stage2Layer2Raw'),
    #cms.InputTag('stage2DemuxRaw')
)

# raw to digi
import EventFilter.L1TRawToDigi.l1tRawToDigi_cfi
process.l1tDigis = EventFilter.L1TRawToDigi.l1tRawToDigi_cfi.l1tRawToDigi.clone()
process.l1tDigis.FedId = cms.int32(2)
process.l1tDigis.InputLabel = cms.InputTag("rawData")

process.l1tDigis.Unpackers = cms.vstring([ "l1t::CaloTowerUnpackerFactory",
                                           "l1t::EGammaUnpackerFactory",
                                           "l1t::EtSumUnpackerFactory",
                                           "l1t::JetUnpackerFactory",
                                           "l1t::TauUnpackerFactory",
                                           "l1t::MPUnpackerFactory"])

### emulator ###

# upgrade calo stage 2
process.load('L1Trigger.L1TCalorimeter.L1TCaloStage2_cff')
process.caloStage2Digis.towerToken = cms.InputTag("l1tDigis")

process.load("L1Trigger.L1TCalorimeter.caloStage2Params_cfi")



### diagnostics ###

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawData"),
    feds = cms.untracked.vint32 ( 1, 2 ),
    dumpPayload = cms.untracked.bool ( True )
)

# plots from unpacker
import L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi
process.rawPlots = L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
process.rawPlots.towerToken = cms.InputTag("l1tDigis")
process.rawPlots.clusterToken = cms.InputTag("None")
process.rawPlots.egToken = cms.InputTag("None")
process.rawPlots.tauToken = cms.InputTag("None")
process.rawPlots.jetToken = cms.InputTag("l1tDigis")
process.rawPlots.etSumToken = cms.InputTag("l1tDigis")

# plots from emulator
process.simPlots = L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
process.simPlots.towerToken = cms.InputTag("caloStage2Digis")
process.simPlots.clusterToken = cms.InputTag("caloStage2Digis")
process.simPlots.egToken = cms.InputTag("caloStage2Digis")
process.simPlots.tauToken = cms.InputTag("caloStage2Digis")
process.simPlots.jetToken = cms.InputTag("caloStage2Digis")
process.simPlots.etSumToken = cms.InputTag("caloStage2Digis")


# Path and EndPath definitions
process.path = cms.Path(

    # produce RAW
    process.stage2Layer2Raw
#    +process.stage2DemuxRaw
    +process.rawData

    # unpack
    +process.l1tDigis

    # emulator
    +process.caloStage2Digis

    # diagnostics
    +process.dumpRaw
    +process.rawPlots
    +process.simPlots
)

process.out = cms.EndPath(
    process.output
)
