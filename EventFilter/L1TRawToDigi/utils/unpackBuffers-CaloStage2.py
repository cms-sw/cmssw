from __future__ import print_function
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
options.register('fwVersion',
                 268501043,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Firmware version for unpacker configuration")
options.register('demuxFWVersion',
                 268501079,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Firmware version for demux unpacker configuration")
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
options.register('mpKeyLinkRx',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "MP packet key link (Rx)")
options.register('mpKeyLinkTx',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "MP packet key link (Tx)")
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
                 29,
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
                 9,#11,
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
process.MessageLogger.L1TCaloEvents=dict()

process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

if (options.dump):
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

mpLatencies = cms.untracked.vint32()
for i in range (0,options.nMP):
    mpLatencies.append(0)

boardIds = cms.untracked.vint32(range(0,options.nMP))

boardOffset = options.skipEvents % options.nMP

gtOffset = options.gtOffset + (options.skipEvents * options.gtFramesPerEvent)

# print some debug info
print("Job config :")
print("maxEvents     = ", options.maxEvents)
print("skipEvents    = ", options.skipEvents)
print(" ")

# MP config
if (options.doMP):
    print("MP config :")
    print("nBoards       = ", options.nMP)
    print("mpBoardOffset = ", boardOffset)
    print("mpOffset      = ", mpOffsets)
    print(" ")

mpBlock = cms.untracked.PSet(
    rxBlockLength    = cms.untracked.vint32(40,40,40,40, # q0 0-3
                                            40,40,40,40, # q1 4-7
                                            40,40,40,40, # q2 8-11
                                            40,40,40,40, # q3 12-15
                                            40,40,40,40, # q4 16-19
                                            40,40,40,40, # q5 20-23
                                            40,40,40,40, # q6 24-27
                                            40,40,40,40, # q7 28-31
                                            40,40,40,40, # q8 32-35
                                            40,40,40,40, # q9 36-39
                                            40,40,40,40, # q10 40-43
                                            40,40,40,40, # q11 44-47
                                            40,40,40,40, # q12 48-51
                                            40,40,40,40, # q13 52-55
                                            40,40,40,40, # q14 56-59
                                            40,40,40,40, # q15 60-63
                                            40,40,40,40, # q16 64-67
                                            40,40,40,40), # q17 68-71

    txBlockLength    = cms.untracked.vint32(0,0,0,0, # q0 0-3
                                            0,0,0,0, # q1 4-7
                                            0,0,0,0, # q2 8-11
                                            0,0,0,0, # q3 12-15
                                            0,0,0,0, # q4 16-19
                                            0,0,0,0, # q5 20-23
                                            0,0,0,0, # q6 24-27
                                            0,0,0,0, # q7 28-31
                                            0,0,0,0, # q8 32-35
                                            0,0,0,0, # q9 36-39
                                            0,0,0,0, # q10 40-43
                                            0,0,0,0, # q11 44-47
                                            0,0,0,0, # q12 48-51
                                            0,0,0,0, # q13 52-55
                                            0,0,0,0, # q14 56-59
                                            11,11,11,11, # q15 60-63
                                            11,11,0,0, # q16 64-67
                                            0,0,0,0) # q17 68-71
)

mpBlocks = cms.untracked.VPSet()

for block in range(0,options.nMP):
        mpBlocks.append(mpBlock)

process.stage2MPRaw.nFramesPerEvent    = cms.untracked.int32(options.mpFramesPerEvent)
process.stage2MPRaw.nFramesOffset    = cms.untracked.vuint32(mpOffsets)
process.stage2MPRaw.nFramesLatency   = cms.untracked.vuint32(mpLatencies)
process.stage2MPRaw.boardOffset    = cms.untracked.int32(boardOffset)
process.stage2MPRaw.rxKeyLink    = cms.untracked.int32(options.mpKeyLinkRx)
process.stage2MPRaw.txKeyLink    = cms.untracked.int32(options.mpKeyLinkTx)
process.stage2MPRaw.boardId = cms.untracked.vint32(boardIds)
process.stage2MPRaw.nHeaderFrames = cms.untracked.int32(options.mpHeaderFrames)
process.stage2MPRaw.rxFile = cms.untracked.string("mp_rx_summary.txt")
process.stage2MPRaw.txFile = cms.untracked.string("mp_tx_summary.txt")
process.stage2MPRaw.blocks = cms.untracked.VPSet(mpBlocks)
process.stage2MPRaw.fwVersion = cms.untracked.int32(options.fwVersion)

# Demux config
if (options.doDemux):
    print("Demux config :")
    print("dmOffset      = ", dmOffset)
    print("dmLatency     = ", options.dmLatency)
    print(" ")

process.stage2DemuxRaw.nFramesPerEvent    = cms.untracked.int32(options.dmFramesPerEvent)
process.stage2DemuxRaw.nFramesOffset    = cms.untracked.vuint32(dmOffset)
# add 1 to demux latency to take account of header, match online definition of latency
process.stage2DemuxRaw.nFramesLatency   = cms.untracked.vuint32(options.dmLatency+1)
process.stage2DemuxRaw.rxFile = cms.untracked.string("demux_rx_summary.txt")
process.stage2DemuxRaw.txFile = cms.untracked.string("demux_tx_summary.txt")
process.stage2DemuxRaw.fwVersion = cms.untracked.int32(options.demuxFWVersion)

# GT config
if (options.doGT):
    print("GT config :")
    print("gtOffset      = ", gtOffset)
    print("gtLatency     = ", options.gtLatency)

process.stage2GTRaw.nFramesPerEvent    = cms.untracked.int32(options.gtFramesPerEvent)
process.stage2GTRaw.nFramesOffset    = cms.untracked.vuint32(gtOffset)
process.stage2GTRaw.nFramesLatency   = cms.untracked.vuint32(options.gtLatency)
process.stage2GTRaw.rxFile = cms.untracked.string("uGT/rx_summary.txt")
process.stage2GTRaw.txFile = cms.untracked.string("uGT/tx_summary.txt")


process.rawDataCollector.verbose = cms.untracked.int32(2)


# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    token = cms.untracked.InputTag("rawDataCollector"),
    feds = cms.untracked.vint32 ( 1360, 1366, 1404 ),
    dumpPayload = cms.untracked.bool ( options.dump )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.caloStage2Digis_cfi')
process.caloStage2Digis.InputLabel = cms.InputTag('rawDataCollector')
process.caloStage2Digis.debug      = cms.untracked.bool(options.debug)
process.caloStage2Digis.FWId  = cms.uint32(options.fwVersion)
process.caloStage2Digis.DmxFWId = cms.uint32(options.demuxFWVersion)
process.caloStage2Digis.FWOverride = cms.bool(True)
process.caloStage2Digis.TMTCheck   = cms.bool(False)

process.load('EventFilter.L1TRawToDigi.gtStage2Digis_cfi')
process.gtStage2Digis.InputLabel = cms.InputTag('rawDataCollector')

# object analyser
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")


# Path and EndPath definitions
process.path = cms.Path(
    process.stage2MPRaw
    +process.stage2DemuxRaw
    +process.stage2GTRaw
    +process.rawDataCollector
    #+process.dumpRaw
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


