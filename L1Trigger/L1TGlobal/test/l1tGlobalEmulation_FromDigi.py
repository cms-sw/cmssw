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

if (options.maxEvents == -1):
    options.maxEvents = 1


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
    fileNames = cms.untracked.vstring("file:l1tCalo_2016_EDM_Save.root") 
)




# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2016_histos.root')


# enable debug message logging for our modules
process.MessageLogger.categories.append('L1TCaloEvents')
process.MessageLogger.categories.append('L1TGlobalEvents')
process.MessageLogger.categories.append('l1t|Global')

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
#process.TriggerMenuXml.DefXmlFile = 'L1Menu_Collisions2015_25nsStage1_v6_uGT_v2.xml'
process.TriggerMenuXml.DefXmlFile = 'L1Menu_Collisions2015_25nsStage1_v6_uGT_v3.xml'

process.load('L1Trigger.L1TGlobal.TriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')

## Run the Stage 2 uGT emulator
process.load('L1Trigger.L1TGlobal.simGtStage2Digis_cff')
process.simGtStage2Digis.caloInputTag = cms.InputTag('caloStage2Digis')
process.simGtStage2Digis.GmtInputTag = cms.InputTag('None')
process.simGtStage2Digis.PrescaleCSVFile = cms.string('prescale_L1TGlobal.csv')
process.simGtStage2Digis.PrescaleSet = cms.uint32(1)



# gt analyzer
process.l1tGlobalAnalyzer = cms.EDAnalyzer('L1TGlobalAnalyzer',
    doText = cms.untracked.bool(False),
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
    emulDxAlgToken = cms.InputTag("None"),
    emulGtAlgToken = cms.InputTag("simGtStage2Digis")
)


# dump records
process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
        egInputTag    = cms.InputTag("gtStage2Digis","GT"),
		muInputTag    = cms.InputTag("gtStage2Digis","GT"),
		tauInputTag   = cms.InputTag(""),
		jetInputTag   = cms.InputTag("gtStage2Digis","GT"),
		etsumInputTag = cms.InputTag("gtStage2Digis","GT"),
		uGtRecInputTag = cms.InputTag(""),
		uGtAlgInputTag = cms.InputTag("simGtStage2Digis"),
		uGtExtInputTag = cms.InputTag(""),
		bxOffset       = cms.int32(0),
		minBx          = cms.int32(0),
		maxBx          = cms.int32(0),
		minBxVec       = cms.int32(0),
		maxBxVec       = cms.int32(0),		
		dumpGTRecord   = cms.bool(True),
        dumpTrigResults= cms.bool(True),
		dumpVectors    = cms.bool(False),
		tvFileName     = cms.string( "TestVector.txt" ),
                psFileName     = cms.string( "prescale_L1TGlobal.csv" ),
                psColumn       = cms.int32(1)
		 )
		 




# Path and EndPath definitions
process.path = cms.Path(
    process.simGtStage2Digis
    +process.l1tGlobalAnalyzer
    +process.dumpGTRecord
)


