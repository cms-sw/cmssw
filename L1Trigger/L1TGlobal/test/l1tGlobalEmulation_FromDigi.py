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
options.register('newXML',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "New XML Grammar")		 
                 
options.parseArguments()

#if (options.maxEvents == -1):
#    options.maxEvents = 1


process = cms.Process('uGTEmulation')

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
    skipEvents=cms.untracked.uint32(options.skipEvents),
    fileNames = cms.untracked.vstring(options.inputFiles) 
)




# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2016_histos.root')


# enable debug message logging for our modules
#
#
#
process.MessageLogger.debugModules = cms.untracked.vstring('simGlobalStage2Digis')
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')

process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


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
   process.TriggerMenuXml.DefXmlFile = 'L1Menu_CollisionsHeavyIons2015_v4_uGT_v2.xml'
   #process.TriggerMenuXml.DefXmlFile = 'MuonTest.xml'


process.load('L1Trigger.L1TGlobal.TriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')

## Run the Stage 2 uGT emulator
process.load('L1Trigger.L1TGlobal.simGlobalStage2Digis_cff')
process.simGlobalStage2Digis.caloInputTag = cms.InputTag("gtStage2Digis","GT")
process.simGlobalStage2Digis.GmtInputTag = cms.InputTag("gtStage2Digis","GT")
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


# dump records
process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
        egInputTag    = cms.InputTag("gtStage2Digis","GT"),
		muInputTag    = cms.InputTag("gtStage2Digis","GT"),
		tauInputTag   = cms.InputTag("gtStage2Digis","GT"),
		jetInputTag   = cms.InputTag("gtStage2Digis","GT"),
		etsumInputTag = cms.InputTag("gtStage2Digis","GT"),
		uGtAlgInputTag = cms.InputTag("gtStage2Digis","GT"),
		uGtExtInputTag = cms.InputTag("gtStage2Digis","GT"),
		bxOffset       = cms.int32(0),
		minBx          = cms.int32(0),
		maxBx          = cms.int32(0),
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
    process.simGlobalStage2Digis
    +process.l1tGlobalAnalyzer
    +process.dumpGTRecord
)


