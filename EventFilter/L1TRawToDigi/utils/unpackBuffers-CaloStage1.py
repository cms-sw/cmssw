# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms


# options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('skipEvents',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events to skip")
options.register('framesPerEvent',
                 6,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "N frames per event")
options.register('offset',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Jet board offset (frames)")
options.register('egLatency',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "EG board latency (frames)")
options.register('jetLatency',
                 40,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Jet board latency (frames)")
options.register('egDelay',
                 54,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "EG input delay wrt regions (frames)")
options.register('dump',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Print RAW data")

                 
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
    fileName = cms.untracked.string('l1tCalo_2015_EDM.root')
)

# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2015_histos.root')


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
process.load('EventFilter.L1TRawToDigi.stage1MP7BufferRaw_cfi')

# skip events
jetOffset = options.offset + (options.skipEvents * options.framesPerEvent)
egOffset  = jetOffset + options.egDelay


# print some debug info
print "egOffset      = ", egOffset
print "jetOffset     = ", jetOffset
print " "


# pack into arrays
latencies = [ options.jetLatency, options.egLatency ]
offsets   = [ jetOffset,  egOffset ]

process.stage1Raw.nFramesPerEvent    = cms.untracked.int32(options.framesPerEvent)
process.stage1Raw.nFramesOffset    = cms.untracked.vuint32(offsets)
process.stage1Raw.nFramesLatency   = cms.untracked.vuint32(latencies)
process.stage1Raw.rxFile = cms.untracked.string("rx_summary.txt")
process.stage1Raw.txFile = cms.untracked.string("tx_summary.txt")

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("stage1Raw"),
    feds = cms.untracked.vint32 ( 1352 ),
    dumpPayload = cms.untracked.bool ( options.dump )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.caloStage1Digis_cfi')
process.caloStage1Digis.InputLabel = cms.InputTag('stage1Raw')

# Path and EndPath definitions
process.path = cms.Path(
    process.stage1Raw
    +process.dumpRaw
    +process.caloStage1Digis

)

process.out = cms.EndPath(
    process.output
)

