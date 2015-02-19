# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms


# options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('mpLatency',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "MP latency (frames)")
options.register('mpOffset',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "MP offset (frames)")
options.register('dmFramesPerEvent',
                 6,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Demux frames per event")
options.register('dmLatency',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Demux latency (frames)")
options.register('dmOffset',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Demux offset (frames)")
options.register('dump',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Print RAW data")
options.register('doMP',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Read MP data")
options.register('doDemux',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Read demux data")
options.register('nMP',
                 11,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of MPs")
                 
options.parseArguments()

if (options.maxEvents == -1):
    options.maxEvents = 1


process = cms.Process('Raw2Digi')

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
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)


# Output definition
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *"),
    fileName = cms.untracked.string('l1tCalo_2016_EDM.root')
)

# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2016_histos.root')


# enable debug message logging for our modules
process.MessageLogger = cms.Service(
    "MessageLogger",
    threshold  = cms.untracked.string('DEBUG'),
    categories = cms.untracked.vstring('L1T'),
    debugModules = cms.untracked.vstring('*')
#        'mp7BufferDumpToRaw',
#        'l1tDigis',
#        'caloStage1Digis'
#    )
)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


# buffer dump to RAW
process.load('EventFilter.L1TRawToDigi.stage2MP7BufferRaw_cff')

# skip events
dmOffset = options.dmOffset + (options.skipEvents * options.dmFramesPerEvent)

mpOffsets = cms.untracked.vint32()
for i in range (0,options.nMP):
    offset = options.mpOffset + (options.skipEvents / options.nMP)
    if (i < options.skipEvents % options.nMP):
        offset = offset + 1    
    mpOffsets.append(offset)

boardOffset = options.skipEvents % options.nMP

# print some debug info
print "nMP           = ", options.nMP
print "maxEvents     = ", options.maxEvents
print "skipEvents    = ", options.skipEvents
print "dmOffset      = ", dmOffset
print "mpBoardOffset = ", boardOffset
print "mpOffset      = ", mpOffsets
print " "

#mpLatencies = cms.untracked.vint32( 0,0,0,0,0,0,0,0,0,0,0 )
#for i in range (0,11):
#    mpLatencies.[i](options.mpLatency)

process.stage2MPRaw.nFramesOffset    = cms.untracked.vuint32(mpOffsets)
process.stage2MPRaw.boardOffset    = cms.untracked.int32(boardOffset)
#process.stage2MPRaw.nFramesLatency   = cms.untracked.vuint32(mpLatencies)
process.stage2MPRaw.rxFile = cms.untracked.string("mp_rx_summary.txt")
process.stage2MPRaw.txFile = cms.untracked.string("mp_tx_summary.txt")

process.stage2DemuxRaw.nFramesPerEvent    = cms.untracked.int32(options.dmFramesPerEvent)
process.stage2DemuxRaw.nFramesOffset    = cms.untracked.vuint32(dmOffset)
process.stage2DemuxRaw.nFramesLatency   = cms.untracked.vuint32(options.dmLatency)
process.stage2DemuxRaw.rxFile = cms.untracked.string("demux_rx_summary.txt")
process.stage2DemuxRaw.txFile = cms.untracked.string("demux_tx_summary.txt")

process.rawDataCollector.verbose = cms.untracked.int32(2)

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawDataCollector"),
    feds = cms.untracked.vint32 ( 1360, 1366 ),
    dumpPayload = cms.untracked.bool ( options.dump )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.caloStage2Digis_cfi')
process.caloStage2Digis.InputLabel = cms.InputTag('rawDataCollector')

process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("caloStage2Digis")
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpEGToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpTauToken = cms.InputTag("None")

# Path and EndPath definitions
process.path = cms.Path(
    process.stage2MP7BufferRaw
#    process.stage2MPRaw
#    +process.stage2DemuxRaw
    +process.dumpRaw
    +process.caloStage2Digis
#    +process.l1tStage2CaloAnalyzer
)

process.out = cms.EndPath(
    process.output
)

