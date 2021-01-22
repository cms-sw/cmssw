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
                 "Read input from streamer file")
options.register('debug',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Enable debug data")
options.register('dumpRaw',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Print RAW data")
options.register('dumpDigis',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Print digis")
options.register('histos',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Produce standard histograms")
options.register('edm',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Produce EDM file")
options.register('valEvents',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Filter on validation events")
options.register('process',
                 '',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Rename process if used")
options.register('mps',
                 '',
                 VarParsing.VarParsing.multiplicity.list,
                 VarParsing.VarParsing.varType.int,
                 "List of MPs to process")
options.register('json',
                 '',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "JSON file with list of good lumi sections")
options.register('evtDisp',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 'Produce histos for individual events')

options.parseArguments()


pname="Raw2Digi"
if (options.process!=""):
    pname=options.process

process = cms.Process(pname)

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
if (options.streamer) :
    process.source = cms.Source(
        "NewEventStreamFileReader",
        fileNames = cms.untracked.vstring (options.inputFiles),
        skipEvents=cms.untracked.uint32(options.skipEvents)
    )
else :
    process.source = cms.Source (
        "PoolSource",
        fileNames = cms.untracked.vstring (options.inputFiles),
        skipEvents=cms.untracked.uint32(options.skipEvents)
    )

if (options.json):
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = options.json).getVLuminosityBlockRange()

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)


# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2016_histos.root')

# enable debug message logging for our modules

process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

if (options.dumpRaw):
    process.MessageLogger.files.infos = cms.untracked.PSet(
        INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        L1TCaloEvents = cms.untracked.PSet(
            limit = cms.untracked.int32(10000)
        )
    )

if (options.debug):
#    process.MessageLogger.debugModules = cms.untracked.vstring('L1TRawToDigi:caloStage2Digis', 'MP7BufferDumpToRaw:stage2MPRaw', 'MP7BufferDumpToRaw:stage2DemuxRaw')
    process.MessageLogger.debugModules = cms.untracked.vstring('*')
    process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

# validation event filter
process.load('EventFilter.L1TRawToDigi.validationEventFilter_cfi')

# MP selectah
process.load('EventFilter.L1TRawToDigi.tmtFilter_cfi')
process.tmtFilter.mpList = cms.untracked.vint32(options.mps)

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    token = cms.untracked.InputTag("rawDataCollector"),
    feds = cms.untracked.vint32 ( 1360, 1366, 1404 ),
    dumpPayload = cms.untracked.bool ( options.dumpRaw )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.caloStage2Digis_cfi')
process.caloStage2Digis.InputLabel = cms.InputTag('rawDataCollector')
if (options.debug):
    process.caloStage2Digis.debug = cms.untracked.bool(True)

process.load('EventFilter.L1TRawToDigi.gtStage2Digis_cfi')
process.gtStage2Digis.InputLabel = cms.InputTag('rawDataCollector')

# object analyser
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.doText       = cms.untracked.bool(options.dumpDigis)
process.l1tStage2CaloAnalyzer.doHistos     = cms.untracked.bool(options.histos)
process.l1tStage2CaloAnalyzer.doEvtDisp    = cms.bool(options.evtDisp)

# Path and EndPath definitions
process.path = cms.Path(
    process.validationEventFilter
    +process.dumpRaw
    +process.caloStage2Digis
    +process.gtStage2Digis
    +process.l1tStage2CaloAnalyzer
)

# enable validation event filtering
if (not options.valEvents):
    process.path.remove(process.validationEventFilter)

# enable validation event filtering
if (len(options.mps)==0):
    process.path.remove(process.tmtFilter)

# enable RAW printout
if (not options.dumpRaw):
    process.path.remove(process.dumpRaw)

# optional EDM file
if (options.edm):
    process.output = cms.OutputModule(
        "PoolOutputModule",
        outputCommands = cms.untracked.vstring("keep *"),
        SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('path')
        ),
        fileName = cms.untracked.string('l1tCalo_2016_EDM.root')
    )

    process.out = cms.EndPath(
        process.output
    )
