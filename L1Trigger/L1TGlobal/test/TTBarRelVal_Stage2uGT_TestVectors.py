#!/usr/bin/env python
from __future__ import print_function
import sys

"""
The parameters can be changed by adding commandline arguments of the form
::

    runGlobalFakeInputProducer.py nevents=-1

The latter can be used to change parameters in crab.
"""

job = 0 #job number
njob = 1 #number of jobs
nevents = 3564 #number of events
rootout = False #whether to produce root file
dump = False #dump python

# Argument parsing
# vvv


if len(sys.argv) == 2 and ':' in sys.argv[1]:
    argv = sys.argv[1].split(':')
else:
    argv = sys.argv[1:]

for arg in argv:
    (k, v) = map(str.strip, arg.split('='))
    if k not in globals():
        raise "Unknown argument '%s'!" % (k,)
    if isinstance(globals()[k], bool):
        globals()[k] = v.lower() in ('y', 'yes', 'true', 't', '1')
    elif isinstance(globals()[k], int):
        globals()[k] = int(v)
    else:
        globals()[k] = v

neventsPerJob = nevents/njob
skip = job * neventsPerJob

if skip>4:
    skip = skip-4
    neventsPerJob = neventsPerJob+4

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
    input = cms.untracked.int32(neventsPerJob)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
	"/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_POSTLS171_V16-v1/00000/0EF13A80-F2FE-E311-9565-003048FFD7D4.root",
	"/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_POSTLS171_V16-v1/00000/389E0C8A-EFFE-E311-86EA-0025905A6088.root",
	"/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_POSTLS171_V16-v1/00000/52F1C37C-F1FE-E311-89EA-00261894394D.root",
	"/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_POSTLS171_V16-v1/00000/545B93FA-F1FE-E311-8414-0025905A497A.root",
	"/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_POSTLS171_V16-v1/00000/5C8B9784-EFFE-E311-A37A-0025905A60B0.root",
	"/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_POSTLS171_V16-v1/00000/5E835FAA-F3FE-E311-8D88-0025905B8596.root",
	"/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_POSTLS171_V16-v1/00000/6C2B9503-F0FE-E311-858F-0025905A612A.root",
	),
    skipEvents = cms.untracked.uint32(skip)
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


process.dumpGT = cms.EDAnalyzer("l1t::GtInputDump",
                egInputTag    = cms.InputTag("caloStage2Digis"),
		muInputTag    = cms.InputTag(""),
		tauInputTag   = cms.InputTag("caloStage2Digis"),
		jetInputTag   = cms.InputTag("caloStage2Digis"),
		etsumInputTag = cms.InputTag("caloStage2Digis"), 
		minBx         = cms.int32(0),
		maxBx         = cms.int32(0)
		 )

process.bxVectorGT = cms.EDProducer("l1t::BXVectorInputProducer",
                egInputTag     = cms.InputTag("caloStage2Digis"),
		muInputTag     = cms.InputTag(""),
		tauInputTag    = cms.InputTag("caloStage2Digis"),
		jetInputTag    = cms.InputTag("caloStage2Digis"),
		etsumInputTag  = cms.InputTag("caloStage2Digis"), 
		bxFirst        = cms.int32(-2),
		bxLast         = cms.int32(2),
		maxMuCand      = cms.uint32(8),
		maxJetCand     = cms.uint32(12),
		maxEGCand      = cms.uint32(12),
		maxTauCand     = cms.uint32(8),				     				     
		jetEtThreshold = cms.double(10),
		tauEtThreshold = cms.double(10),
		egEtThreshold  = cms.double(10),
		muEtThreshold  = cms.double(1),
		emptyBxTrailer = cms.int32(5),
		emptyBxEvt     = cms.int32(neventsPerJob)
		 )


process.simL1uGtDigis = cms.EDProducer("L1TGlobalProducer",
    #TechnicalTriggersUnprescaled = cms.bool(False),
    ProduceL1GtObjectMapRecord = cms.bool(True),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(1),
    L1DataBxInEvent = cms.int32(5),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    GmtInputTag = cms.InputTag("gtInput"),
    caloInputTag = cms.InputTag("bxVectorGT"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    #WritePsbL1GtDaqRecord = cms.bool(True),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(0)
)

process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
                egInputTag    = cms.InputTag("bxVectorGT"),
		muInputTag    = cms.InputTag("gtInput"),
		tauInputTag   = cms.InputTag("bxVectorGT"),
		jetInputTag   = cms.InputTag("bxVectorGT"),
		etsumInputTag = cms.InputTag("bxVectorGT"),
		uGtRecInputTag = cms.InputTag("simL1uGtDigis"),
		uGtAlgInputTag = cms.InputTag("simL1uGtDigis"),
		uGtExtInputTag = cms.InputTag("simL1uGtDigis"),
		bxOffset       = cms.int32(skip),
		minBx          = cms.int32(-2),
		maxBx          = cms.int32(2),
		minBxVec       = cms.int32(0),
		maxBxVec       = cms.int32(0),		
		dumpGTRecord   = cms.bool(True),
		dumpVectors    = cms.bool(True),
		tvFileName     = cms.string( ("TestVector_%03d.txt") % job )
		 )
		 


# Raw to digi
process.load('Configuration.StandardSequences.RawToDigi_cff')

# upgrade calo stage 2
process.load('L1Trigger.L1TCalorimeter.L1TCaloStage2_PPFromRaw_cff')
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
#process.load('L1Trigger.L1TCalorimeter.l1tCaloAnalyzer_cfi')

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')

# Path and EndPath definitions
process.L1simulation_step = cms.Path(
    process.ecalDigis
    +process.hcalDigis
    +process.L1TCaloStage2_PPFromRaw
#    +process.dumpGT
    +process.bxVectorGT
    +process.simL1uGtDigis
    +process.dumpGTRecord    
    )

process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.L1simulation_step,
                                process.RECOSIMoutput_step)

if dump:
    outfile = open('dump_TTBarRelVal_Stage2uGT_TestVectors_'+repr(job)+'.py','w')
    print(process.dumpPython(), file=outfile)
    outfile.close()
