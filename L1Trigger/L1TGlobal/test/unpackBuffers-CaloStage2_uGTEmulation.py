from __future__ import print_function
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
#
#  This was adapted from unpackBuffers-CaloStage2.py.
#    -Unpacking of the uGT raw data has been added
#    -uGT Emulation starting with the Demux output and/or the uGT input 
#    -Analysis for uGT objects using L1TGlobalAnalyzer
#
#   Brian Winer, March 16, 2015
#  
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
                 False,
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
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Read GT data")
options.register('newXML',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "New XML Grammar")		 
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
process.TFileService.fileName = cms.string('l1tCalo_2016_histos_'+repr(options.gtOffset)+'-'+repr(options.gtLatency)+'.root')


# enable debug message logging for our modules
process.MessageLogger.categories.append('L1TCaloEvents')
process.MessageLogger.categories.append('L1TGlobalEvents')
process.MessageLogger.categories.append('Global')

process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

if (options.dump):
    process.MessageLogger.destinations.append('infos')
    process.MessageLogger.infos = cms.untracked.PSet(
        INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        L1TCaloEvents = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
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

process.stage2MPRaw.nFramesPerEvent    = cms.untracked.int32(options.mpFramesPerEvent)
process.stage2MPRaw.nFramesOffset    = cms.untracked.vuint32(mpOffsets)
process.stage2MPRaw.boardOffset    = cms.untracked.int32(boardOffset)
#process.stage2MPRaw.nFramesLatency   = cms.untracked.vuint32(mpLatencies)
process.stage2MPRaw.rxFile = cms.untracked.string("merge/rx_summary.txt")
process.stage2MPRaw.txFile = cms.untracked.string("merge/tx_summary.txt")

# Demux config
if (options.doDemux):
    print("Demux config :")
    print("dmOffset      = ", dmOffset)
    print("dmLatency     = ", options.dmLatency)
    print(" ")

process.stage2DemuxRaw.nFramesPerEvent    = cms.untracked.int32(options.dmFramesPerEvent)
process.stage2DemuxRaw.nFramesOffset    = cms.untracked.vuint32(dmOffset)
process.stage2DemuxRaw.nFramesLatency   = cms.untracked.vuint32(options.dmLatency)
process.stage2DemuxRaw.rxFile = cms.untracked.string("good/demux/rx_summary.txt")
process.stage2DemuxRaw.txFile = cms.untracked.string("good/demux/tx_summary.txt")

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
    label = cms.untracked.string("rawDataCollector"),
    feds = cms.untracked.vint32 ( 1360, 1366, 1404 ),
    dumpPayload = cms.untracked.bool ( True )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.caloStage2Digis_cfi')
process.caloStage2Digis.InputLabel = cms.InputTag('rawDataCollector')

process.load('EventFilter.L1TRawToDigi.gtStage2Digis_cfi')
process.gtStage2Digis.InputLabel = cms.InputTag('rawDataCollector')


# Setup for Emulation of uGT
process.load('L1Trigger.L1TGlobal.StableParametersConfig_cff')
process.load('L1Trigger.L1TGlobal.TriggerMenuXml_cfi')
process.TriggerMenuXml.TriggerMenuLuminosity = 'startup'
#process.TriggerMenuXml.DefXmlFile = 'L1Menu_CaloSliceTest_2015_v4.xml'
process.TriggerMenuXml.DefXmlFile = 'L1Menu_Point5IntegrationTest_2015_v2.xml'
process.TriggerMenuXml.newGrammar = cms.bool(options.newXML)
if(options.newXML):
   print("Using new XML Grammar ")
   #process.TriggerMenuXml.DefXmlFile = 'L1Menu_Point5IntegrationTest_2015_v1a.xml'
   process.TriggerMenuXml.DefXmlFile = 'MuonTest.xml'

process.load('L1Trigger.L1TGlobal.TriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')

process.emL1uGtFromGtInput = cms.EDProducer("L1TGlobalProducer",
    ProduceL1GtObjectMapRecord = cms.bool(False),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(1),
    L1DataBxInEvent = cms.int32(1),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    GmtInputTag = cms.InputTag("gtStage2Digis","GT"),
    caloInputTag = cms.InputTag("gtStage2Digis","GT"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(5)
)

process.emL1uGtFromDemuxOutput = cms.EDProducer("L1TGlobalProducer",
    ProduceL1GtObjectMapRecord = cms.bool(False),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(1),
    L1DataBxInEvent = cms.int32(1),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    GmtInputTag = cms.InputTag(""),
    caloInputTag = cms.InputTag("caloStage2Digis"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(5)
)



# object analyser
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.doText = cms.untracked.bool(options.debug)
process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("caloStage2Digis")
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpEGToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpTauToken = cms.InputTag("None")


# gt analyzer
process.l1tGlobalAnalyzer = cms.EDAnalyzer('L1TGlobalAnalyzer',
    doText = cms.untracked.bool(options.debug),
    dmxEGToken = cms.InputTag("caloStage2Digis"),
    dmxTauToken = cms.InputTag("None"),
    dmxJetToken = cms.InputTag("caloStage2Digis"),
    dmxEtSumToken = cms.InputTag("caloStage2Digis"),
    muToken = cms.InputTag("gtStage2Digis","GT"),
    egToken = cms.InputTag("gtStage2Digis","GT"),
    tauToken = cms.InputTag("None"),
    jetToken = cms.InputTag("gtStage2Digis","GT"),
    etSumToken = cms.InputTag("gtStage2Digis","GT"),
    gtAlgToken = cms.InputTag("gtStage2Digis"),
    emulDxAlgToken = cms.InputTag("emL1uGtFromDemuxOutput"),
    emulGtAlgToken = cms.InputTag("emL1uGtFromGtInput")
)


# dump records
process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
                egInputTag    = cms.InputTag("gtStage2Digis","GT"),
		muInputTag    = cms.InputTag("gtStage2Digis","GT"),
		tauInputTag   = cms.InputTag(""),
		jetInputTag   = cms.InputTag("gtStage2Digis","GT"),
		etsumInputTag = cms.InputTag("gtStage2Digis","GT"),
		uGtRecInputTag = cms.InputTag(""),
		uGtAlgInputTag = cms.InputTag("emL1uGtFromGtInput"),
		uGtExtInputTag = cms.InputTag(""),
		bxOffset       = cms.int32(0),
		minBx          = cms.int32(0),
		maxBx          = cms.int32(0),
		minBxVec       = cms.int32(0),
		maxBxVec       = cms.int32(0),		
		dumpGTRecord   = cms.bool(True),
		dumpVectors    = cms.bool(True),
		tvFileName     = cms.string( "TestVector.txt" )
		 )
		 




# Path and EndPath definitions
process.path = cms.Path(
#    process.stage2MPRaw
     process.stage2DemuxRaw
    +process.stage2GTRaw
    +process.rawDataCollector
    +process.dumpRaw
    +process.caloStage2Digis
    +process.gtStage2Digis
    +process.emL1uGtFromGtInput
#    +process.emL1uGtFromDemuxOutput
#    +process.l1tStage2CaloAnalyzer
#    +process.l1tGlobalAnalyzer
#    +process.dumpGTRecord
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

