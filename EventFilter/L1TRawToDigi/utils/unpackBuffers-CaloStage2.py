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
options.register('mpFramesPerEvent',
                 40,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "MP frames per event")
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
options.register('mpHeaderFrames',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "MP header frames in tx")
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
                 28,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Demux offset (frames)")
options.register('gtFramesPerEvent',
                 6,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "GT frames per event")
options.register('gtLatency',
                 47,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "GT latency (frames)")
options.register('gtOffset',
                 15,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "GT offset (frames)")
options.register('dump',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Print RAW data")
options.register('debug',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Enable debug data")
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
options.register('doGT',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Read GT data")
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
process.MessageLogger.categories.append('L1TCaloEvents')

process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

if (options.dump):
    process.MessageLogger.infos.placeholder = cms.untracked.bool(False)
    process.MessageLogger.infos.INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
    process.MessageLogger.infos.L1TCaloEvents = cms.untracked.PSet(
      optionalPSet = cms.untracked.bool(True),
      limit = cms.untracked.int32(10000)
    )

if (options.debug):
#    process.MessageLogger.debugModules = cms.untracked.vstring('L1TRawToDigi:caloStage2Digis', 'MP7BufferDumpToRaw:stage2MPRaw', 'MP7BufferDumpToRaw:stage2DemuxRaw')
    process.MessageLogger.debugModules = cms.untracked.vstring('*')
    process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')



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

gtOffset = options.gtOffset + (options.skipEvents * options.gtFramesPerEvent)

# print some debug info
print "Job config :"
print "maxEvents     = ", options.maxEvents
print "skipEvents    = ", options.skipEvents
print " "

# MP config
if (options.doMP):
    print "MP config :"
    print "nBoards       = ", options.nMP
    print "mpBoardOffset = ", boardOffset
    print "mpOffset      = ", mpOffsets
    print " "

process.stage2MPRaw.nFramesPerEvent    = cms.untracked.int32(options.mpFramesPerEvent)
process.stage2MPRaw.nFramesOffset    = cms.untracked.vuint32(mpOffsets)
process.stage2MPRaw.boardOffset    = cms.untracked.int32(boardOffset)
#process.stage2MPRaw.nFramesLatency   = cms.untracked.vuint32(mpLatencies)
process.stage2MPRaw.nHeaderFrames = cms.untracked.int32(options.mpHeaderFrames)
process.stage2MPRaw.rxFile = cms.untracked.string("mp_rx_summary.txt")
process.stage2MPRaw.txFile = cms.untracked.string("mp_tx_summary.txt")

# Demux config
if (options.doDemux):
    print "Demux config :"
    print "dmOffset      = ", dmOffset
    print "dmLatency     = ", options.dmLatency
    print " "

process.stage2DemuxRaw.nFramesPerEvent    = cms.untracked.int32(options.dmFramesPerEvent)
process.stage2DemuxRaw.nFramesOffset    = cms.untracked.vuint32(dmOffset)
process.stage2DemuxRaw.nFramesLatency   = cms.untracked.vuint32(options.dmLatency)
process.stage2DemuxRaw.rxFile = cms.untracked.string("demux_rx_summary.txt")
process.stage2DemuxRaw.txFile = cms.untracked.string("demux_tx_summary.txt")

# GT config
if (options.doGT):
    print "GT config :"
    print "gtOffset      = ", gtOffset
    print "gtLatency     = ", options.gtLatency

process.stage2GTRaw.nFramesPerEvent    = cms.untracked.int32(options.gtFramesPerEvent)
process.stage2GTRaw.nFramesOffset    = cms.untracked.vuint32(gtOffset)
process.stage2GTRaw.nFramesLatency   = cms.untracked.vuint32(options.gtLatency)
process.stage2GTRaw.rxFile = cms.untracked.string("uGT/rx_summary.txt")
process.stage2GTRaw.txFile = cms.untracked.string("uGT/tx_summary.txt")


process.rawDataCollector.verbose = cms.untracked.int32(2)


# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawDataCollector"),
    feds = cms.untracked.vint32 ( 1360, 1366, 1404 ),
    dumpPayload = cms.untracked.bool ( False )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.caloStage2Digis_cfi')
process.caloStage2Digis.InputLabel = cms.InputTag('rawDataCollector')

process.load('EventFilter.L1TRawToDigi.gtStage2Digis_cfi')
process.gtStage2Digis.InputLabel = cms.InputTag('rawDataCollector')

# object analyser
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("caloStage2Digis")
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpEGToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpTauToken = cms.InputTag("None")

# Path and EndPath definitions
process.path = cms.Path(
    process.stage2MPRaw
    +process.stage2DemuxRaw
    +process.stage2GTRaw
    +process.rawDataCollector
    +process.dumpRaw
    +process.caloStage2Digis
    +process.gtStage2Digis
    +process.l1tStage2CaloAnalyzer
)

if (not options.doMP):
    process.path.remove(process.stage2MPRaw)

if (not options.doDemux):
    process.path.remove(process.stage2DemuxRaw)

if (not options.doGT):
    process.path.remove(process.stage2GTRaw)

process.out = cms.EndPath(
    process.output
)

