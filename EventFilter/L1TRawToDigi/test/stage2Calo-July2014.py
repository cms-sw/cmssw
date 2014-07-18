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
    input = cms.untracked.int32(21)
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
process.stage2Layer2Raw.fedId     = cms.untracked.int32(2)
process.stage2Layer2Raw.rxFile    = cms.untracked.string("rx_summary.txt")
process.stage2Layer2Raw.txFile    = cms.untracked.string("tx_summary.txt")
process.stage2Layer2Raw.txLatency = cms.untracked.int32(54)
process.stage2Layer2Raw.rxBlockLength    = cms.untracked.vint32(
    41,41,41,41,41,41,41,41,41,
    41,41,41,41,41,41,41,41,41,
    41,41,41,41,41,41,41,41,41,
    41,41,41,41,41,41,41,41,41,
    41,41,41,41,41,41,41,41,41,
    41,41,41,41,41,41,41,41,41,
    41,41,41,41,41,41,41,41,41,
    41,41,41,41,41,41,41,41,41)
process.stage2Layer2Raw.txBlockLength    = cms.untracked.vint32(
    39,39,39,39,39,39,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0)

# raw data from Demux
process.stage2DemuxRaw = EventFilter.L1TRawToDigi.mp7BufferDumpToRaw_cfi.mp7BufferDumpToRaw.clone()
process.stage2DemuxRaw.fedId = cms.untracked.int32(1)
process.stage2DemuxRaw.rxFile = cms.untracked.string("")
process.stage2DemuxRaw.txFile = cms.untracked.string("tx_summary.txt")
process.stage2DemuxRaw.txLatency = cms.untracked.int32(54)
process.stage2DemuxRaw.rxBlockLength    = cms.untracked.vint32(
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0)
process.stage2DemuxRaw.txBlockLength    = cms.untracked.vint32(
    12,4,12,8,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0)

# merge raw data
import EventFilter.RawDataCollector.rawDataCollector_cfi
process.rawData = EventFilter.RawDataCollector.rawDataCollector_cfi.rawDataCollector.clone()
process.rawData.RawCollectionList = cms.VInputTag(
    cms.InputTag('stage2Layer2Raw'),
    cms.InputTag('stage2DemuxRaw')
)

# raw to digi
import EventFilter.L1TRawToDigi.l1tRawToDigi_cfi
process.l1tDigis = EventFilter.L1TRawToDigi.l1tRawToDigi_cfi.l1tRawToDigi.clone()
process.l1tDigis.FedId = cms.int32(2)
process.l1tDigis.InputLabel = cms.InputTag("rawData")


#diagnostics

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawData"),
    feds = cms.untracked.vint32 ( 1, 2 ),
    dumpPayload = cms.untracked.bool ( True )
)


process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("l1tDigis")
#process.l1tStage2CaloAnalyzer.towerPreCompressionToken = cms.InputTag("l1tDigis")
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.egToken = cms.InputTag("l1tDigis")
process.l1tStage2CaloAnalyzer.tauToken = cms.InputTag("l1tDigis")
process.l1tStage2CaloAnalyzer.jetToken = cms.InputTag("l1tDigis")
process.l1tStage2CaloAnalyzer.etSumToken = cms.InputTag("l1tDigis")

# emulator

# upgrade calo stage 2
#process.load('L1Trigger.L1TCalorimeter.L1TCaloStage2_cff')
#process.caloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
#process.caloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")



# Path and EndPath definitions
process.path = cms.Path(
    process.stage2Layer2Raw
    +process.stage2DemuxRaw
    +process.rawData
    +process.l1tDigis
    +process.dumpRaw
    +process.l1tStage2CaloAnalyzer
)

process.out = cms.EndPath(
    process.output
)
