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
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

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
        'stage2Layer2Raw',
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
from EventFilter.L1TRawToDigi.mp7BufferDumpToRaw_cfi import mp7BufferDumpToRaw
process.stage2Layer2Raw = mp7BufferDumpToRaw.clone()
process.stage2Layer2Raw.fedId = cms.untracked.int32(2)
process.stage2Layer2Raw.rxFile = cms.untracked.string("rx_summary.txt")
process.stage2Layer2Raw.txFile = cms.untracked.string("tx_summary.txt")

# raw data from Demux
process.stage2DemuxRaw = mp7BufferDumpToRaw.clone()
process.stage2DemuxRaw.fedId = cms.untracked.int32(1)
process.stage2DemuxRaw.rxFile = cms.untracked.string("")
process.stage2DemuxRaw.txFile = cms.untracked.string("")

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("stage2Layer2Raw"),
    feds = cms.untracked.vint32 ( 2 ),
    dumpPayload = cms.untracked.bool ( True )
)

# raw to digi
import EventFilter.L1TRawToDigi.l1tRawToDigi_cfi
process.l1tDigis = EventFilter.L1TRawToDigi.l1tRawToDigi_cfi.l1tRawToDigi.clone()
process.l1tDigis.FedId = cms.int32(2)
process.l1tDigis.InputLabel = cms.InputTag("stage2Layer2Raw")
process.l1tDigis.Unpackers = cms.vstring(["l1t::CaloTowerUnpackerFactory",
                                          "l1t::EGammaUnpackerFactory",
                                          "l1t::EtSumUnpackerFactory",
                                          "l1t::JetUnpackerFactory",
                                          "l1t::TauUnpackerFactory",
                                          "l1t::MPUnpackerFactory"])

# upgrade calo stage 2
#process.load('L1Trigger.L1TCalorimeter.L1TCaloStage2_cff')
#process.caloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
#process.caloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")

process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("l1tDigis")
#process.l1tStage2CaloAnalyzer.towerPreCompressionToken = cms.InputTag("l1tDigis")
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.egToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.tauToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.jetToken = cms.InputTag("l1tDigis")
process.l1tStage2CaloAnalyzer.etSumToken = cms.InputTag("l1tDigis")

# Path and EndPath definitions
process.path = cms.Path(
    process.stage2Layer2Raw
    +process.dumpRaw
    +process.l1tDigis
    +process.l1tStage2CaloAnalyzer
)

process.out = cms.EndPath(
    process.output
)
