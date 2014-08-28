#!/usr/bin/env python
import sys


import FWCore.ParameterSet.Config as cms

process = cms.Process('L1')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_POSTLS171_V16-v1/00000/0EF13A80-F2FE-E311-9565-003048FFD7D4.root") 
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('l1 nevts:1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *"),#process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('L1.root'),
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
    destinations   = cms.untracked.vstring(
	'detailedInfo',
	'critical'
    ),
    detailedInfo   = cms.untracked.PSet(
	threshold  = cms.untracked.string('DEBUG') 
    ),
    debugModules = cms.untracked.vstring(
	'l1tCaloStage2Layer1Digis',
	'l1tCaloStage2Digis'
    )
)

## Load our L1 menu
process.load('L1Trigger.L1TGlobal.StableParametersConfig_cff')
process.load('L1Trigger.L1TGlobal.TriggerMenuXml_cfi')
process.TriggerMenuXml.TriggerMenuLuminosity = 'startup'
process.TriggerMenuXml.DefXmlFile = 'L1Menu_Reference_2014.xml'

process.load('L1Trigger.L1TGlobal.TriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')

process.simL1uGtDigis = cms.EDProducer("l1t::GtProducer",
    ProduceL1GtObjectMapRecord = cms.bool(True),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(1),
    L1DataBxInEvent = cms.int32(1),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    GmtInputTag = cms.InputTag(""),
    caloInputTag = cms.InputTag("caloStage2Digis"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(0)
)

process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
    egInputTag     = cms.InputTag("caloStage2Digis"),
    muInputTag     = cms.InputTag(""),
    tauInputTag    = cms.InputTag("caloStage2Digis"),
    jetInputTag    = cms.InputTag("caloStage2Digis"),
    etsumInputTag  = cms.InputTag("caloStage2Digis"),
    uGtRecInputTag = cms.InputTag("simL1uGtDigis"),
    uGtAlgInputTag = cms.InputTag("simL1uGtDigis"),
    uGtExtInputTag = cms.InputTag("simL1uGtDigis"),
    bxOffset	   = cms.int32(0),
    minBx	   = cms.int32(0),
    maxBx	   = cms.int32(0),
    minBxVec	   = cms.int32(0),
    maxBxVec	   = cms.int32(0),	    
    dumpGTRecord   = cms.bool(True),
    dumpVectors    = cms.bool(False),
    tvFileName     = cms.string( "uGT_TestVector.txt")
)
		 


# Raw to digi
process.load('Configuration.StandardSequences.RawToDigi_cff')

# upgrade calo stage 2
process.load('L1Trigger.L1TCalorimeter.L1TCaloStage2_PPFromRaw_cff')
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')

# Path and EndPath definitions
process.L1simulation_step = cms.Path(
    process.ecalDigis
    +process.hcalDigis
    +process.L1TCaloStage2_PPFromRaw
    +process.simL1uGtDigis
    +process.dumpGTRecord    
    )

process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.L1simulation_step,
                                process.RECOSIMoutput_step)

