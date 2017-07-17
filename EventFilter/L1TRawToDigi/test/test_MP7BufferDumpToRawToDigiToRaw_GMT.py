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
options.register('streamer',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Use streamer file as input")

options.parseArguments()

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
    input = cms.untracked.int32(10)
)

# Input source
if (options.streamer) :
    process.source = cms.Source(
        "NewEventStreamFileReader",
        fileNames = cms.untracked.vstring (options.inputFiles),
        skipEvents=cms.untracked.uint32(options.skipEvents)
    )
else :
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
process.MessageLogger = cms.Service(
    "MessageLogger",
    threshold  = cms.untracked.string('DEBUG'),
    categories = cms.untracked.vstring('L1T'),
#    l1t   = cms.untracked.PSet(
#	threshold  = cms.untracked.string('DEBUG')
#    ),
    debugModules = cms.untracked.vstring('*'),
#        'stage2GMTRaw',
#    ),
#    cout = cms.untracked.PSet(
#    )
)

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')


# user stuff

# raw data from MP card
process.load('EventFilter.L1TRawToDigi.stage2GMTMP7BufferRaw_cfi')
process.stage2GMTRaw.nFramesLatency   = cms.untracked.vuint32(1)
process.stage2GMTRaw.nFramesOffset   = cms.untracked.vuint32(5)
process.stage2GMTRaw.rxFile = cms.untracked.string("many_events.txt")
process.stage2GMTRaw.txFile = cms.untracked.string("many_events_out.txt")

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("stage2GMTRaw"),
    feds = cms.untracked.vint32(1402),
    dumpPayload = cms.untracked.bool ( True )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.gmtStage2Digis_cfi')
if options.streamer:
    process.gmtStage2Digis.InputLabel = cms.InputTag('rawDataCollector')
    process.dumpRaw.label = cms.untracked.string('rawDataCollector')
else:
    process.gmtStage2Digis.InputLabel = cms.InputTag('stage2GMTRaw')

#process.gmtStage2Digis.FWOverride = cms.bool(True)
#process.gmtStage2Digis.FWId       = cms.uint32(0xffffffff)
process.gmtStage2Digis.debug      = cms.untracked.bool (True)

process.load('EventFilter.L1TRawToDigi.gmtStage2Raw_cfi')

process.dumpRaw2 = process.dumpRaw.clone()
process.dumpRaw2.label = cms.untracked.string("gmtStage2Raw")

# Path and EndPath definitions
process.path = cms.Path(
    process.stage2GMTRaw
    +process.dumpRaw
    +process.gmtStage2Digis
    +process.gmtStage2Raw
    +process.dumpRaw2
)
if options.streamer:
    process.path.remove(process.stage2GMTRaw)

process.out = cms.EndPath(
    process.output
)
