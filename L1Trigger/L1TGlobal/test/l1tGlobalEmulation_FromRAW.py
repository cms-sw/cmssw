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
options.register('streamer',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Read input from streamer file")
options.register('debug',
                 False,
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
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Produce standard histograms")
options.register('edm',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Produce EDM file")
options.register('valEvents',
                 False,
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
options.register('newXML',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "New XML Grammar")		 
                 
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
process.MessageLogger.categories.append('L1TCaloEvents')

process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

if (options.dumpRaw):
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


## Load our L1 menu
process.load('L1Trigger.L1TGlobal.StableParametersConfig_cff')

process.load('L1Trigger.L1TGlobal.TriggerMenuXml_cfi')
process.TriggerMenuXml.TriggerMenuLuminosity = 'startup'
#process.TriggerMenuXml.DefXmlFile = 'L1_Example_Menu_2013.xml'
#process.TriggerMenuXml.DefXmlFile = 'L1Menu_Reference_2014.xml'
process.TriggerMenuXml.DefXmlFile = 'L1Menu_Collisions2015_25nsStage1_v6_uGT_v2a.xml'
#process.TriggerMenuXml.DefXmlFile = 'L1Menu_Collisions2015_25nsStage1_v6_uGT_v3.xml'
process.TriggerMenuXml.newGrammar = cms.bool(options.newXML)
if(options.newXML):
   print("Using new XML Grammar ")
   #process.TriggerMenuXml.DefXmlFile = 'L1Menu_CollisionsHeavyIons2015_v4_uGT_v2.xml'
   #process.TriggerMenuXml.DefXmlFile = 'MuonTest.xml'
   process.TriggerMenuXml.DefXmlFile = 'test_ext.xml'


process.load('L1Trigger.L1TGlobal.TriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')

## Run the Stage 2 uGT emulator
process.load('L1Trigger.L1TGlobal.simGlobalStage2Digis_cff')
process.simGlobalStage2Digis.caloInputTag = cms.InputTag("gtStage2Digis","GT")
process.simGlobalStage2Digis.GmtInputTag = cms.InputTag("gtStage2Digis","GT")
process.simGlobalStage2Digis.extInputTag = cms.InputTag("gtStage2Digis","GT")
process.simGlobalStage2Digis.PrescaleCSVFile = cms.string('prescale_L1TGlobal.csv')
process.simGlobalStage2Digis.PrescaleSet = cms.uint32(1)
process.simGlobalStage2Digis.Verbosity = cms.untracked.int32(0)



# gt analyzer
process.l1tGlobalAnalyzer = cms.EDAnalyzer('L1TGlobalAnalyzer',
    doText = cms.untracked.bool(False),
    dmxEGToken = cms.InputTag("None"),
    dmxTauToken = cms.InputTag("None"),
    dmxJetToken = cms.InputTag("None"),
    dmxEtSumToken = cms.InputTag("None"),
    muToken = cms.InputTag("gtStage2Digis","GT"),
    egToken = cms.InputTag("gtStage2Digis","GT"),
    tauToken = cms.InputTag("gtStage2Digis","GT"),
    jetToken = cms.InputTag("gtStage2Digis","GT"),
    etSumToken = cms.InputTag("gtStage2Digis","GT"),
    gtAlgToken = cms.InputTag("gtStage2Digis","GT"),
    emulDxAlgToken = cms.InputTag("None"),
    emulGtAlgToken = cms.InputTag("simGlobalStage2Digis")
)

# Take Digis to RAW
#process.load("EventFilter.L1TRawToDigi.gtStage2Raw_cfi")
#process.gtStage2Raw.InputLabel = cms.InputTag("gtStage2Digis","GT")


# dump records
process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
        egInputTag    = cms.InputTag("gtStage2Digis","GT"),
		muInputTag    = cms.InputTag("gtStage2Digis","GT"),
		tauInputTag   = cms.InputTag("gtStage2Digis","GT"),
		jetInputTag   = cms.InputTag("gtStage2Digis","GT"),
		etsumInputTag = cms.InputTag("gtStage2Digis","GT"),
		uGtAlgInputTag = cms.InputTag("simGlobalStage2Digis"),
		uGtExtInputTag = cms.InputTag("gtStage2Digis","GT"),
		bxOffset       = cms.int32(0),
		minBx          = cms.int32(-2),
		maxBx          = cms.int32(2),
		minBxVec       = cms.int32(0),
		maxBxVec       = cms.int32(0),		
		dumpGTRecord   = cms.bool(False),
                dumpTrigResults= cms.bool(True),
		dumpVectors    = cms.bool(False),
		tvFileName     = cms.string( "TestVector_Data.txt" ),
                psFileName     = cms.string( "prescale_L1TGlobal.csv" ),
                psColumn       = cms.int32(1)
		 )



# Path and EndPath definitions
process.path = cms.Path(
    process.validationEventFilter
    +process.dumpRaw
    +process.caloStage2Digis
    +process.gtStage2Digis
    +process.simGlobalStage2Digis
    +process.l1tGlobalAnalyzer
    +process.dumpGTRecord 
#    +process.gtStage2Raw   
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



