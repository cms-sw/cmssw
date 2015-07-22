# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
#
#  This was adapted from unpackBuffers-CaloStage2.py.
#    -Dropped formation of RAW data from MP7 Buffer txt files.
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
                 
options.parseArguments()

if (options.maxEvents == -1):
    options.maxEvents = 1


process = cms.Process('Raw2DigiuGTEmul')

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
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
#    inputCommands = cms.untracked.vstring("drop BXVector*_*_*_*"),
    fileNames = cms.untracked.vstring("file:l1tCalo_Test.root"),
    skipEvents = cms.untracked.uint32(options.skipEvents) 
)



# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t_histos.root')


# enable debug message logging for our modules
process.MessageLogger.categories.append('L1TCaloEvents')
process.MessageLogger.categories.append('L1TGlobalEvents')

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

# print some debug info
print "Job config :"
print "maxEvents     = ", options.maxEvents
print "skipEvents    = ", options.skipEvents
print "dump          = ", options.dump
print "debug         = ", options.debug
print " "


# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawDataCollector"),
    feds = cms.untracked.vint32 ( 1360, 1366, 1404 ),
    dumpPayload = cms.untracked.bool ( options.dump )
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
process.TriggerMenuXml.DefXmlFile = 'L1Menu_CaloSliceTest_2015.xml'

process.load('L1Trigger.L1TGlobal.TriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')

process.emL1uGtFromGtInput = cms.EDProducer("l1t::GtProducer",
    ProduceL1GtObjectMapRecord = cms.bool(False),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(1),
    L1DataBxInEvent = cms.int32(1),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    GmtInputTag = cms.InputTag(""),
    caloInputTag = cms.InputTag("gtStage2Digis"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(5)
)

process.emL1uGtFromDemuxOutput = cms.EDProducer("l1t::GtProducer",
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
    dmxEGToken = cms.InputTag("None"),
    dmxTauToken = cms.InputTag("None"),
    dmxJetToken = cms.InputTag("caloStage2Digis"),
    dmxEtSumToken = cms.InputTag("caloStage2Digis"),
    egToken = cms.InputTag("None"),
    tauToken = cms.InputTag("None"),
    jetToken = cms.InputTag("None"),
    etSumToken = cms.InputTag("None"),
    gtAlgToken = cms.InputTag("gtStage2Digis"),
    emulDxAlgToken = cms.InputTag("emL1uGtFromDemuxOutput"),
    emulGtAlgToken = cms.InputTag("None")
)


# Path and EndPath definitions
process.path = cms.Path(
     process.dumpRaw
    +process.caloStage2Digis
    +process.gtStage2Digis
#    +process.emL1uGtFromGtInput
    +process.emL1uGtFromDemuxOutput
    +process.l1tStage2CaloAnalyzer
    +process.l1tGlobalAnalyzer
)



