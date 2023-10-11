#!/usr/bin/env python
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
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('SingleElectronPt10_cfi.py nevts:1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *",
					   "drop *_mix_*_*"),
    fileName = cms.untracked.string('SingleElectronPt10_cfi_py_GEN_SIM_DIGI_L1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(25.01),
        MinPt = cms.double(24.99),
        PartID = cms.vint32(11),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single electron pt 25'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)

# upgrade calo stage 2
process.load('L1Trigger.L1TCalorimeter.caloStage2Params_cfi')
process.load('L1Trigger.L1TCalorimeter.L1TCaloStage2_cff')
process.caloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
process.caloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")
process.esTest = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1TCaloParamsRcd'),
   data = cms.vstring('l1tCaloParams'))),
   verbose = cms.untracked.bool(True)
				)
	    

process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')

process.load('L1Trigger.L1TCalorimeter.l1tStage2InputPatternWriter_cfi')

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
	'caloStage2TowerDigis',
	'caloStage2Digis'
    )
)


## Load our L1 menu
process.load('L1Trigger.L1TGlobal.StableParametersConfig_cff')
process.load('L1Trigger.L1TGlobal.TriggerMenuXml_cfi')
process.TriggerMenuXml.TriggerMenuLuminosity = 'startup'
process.TriggerMenuXml.DefXmlFile = 'L1Menu_Reference_2014.xml'

process.load('L1Trigger.L1TGlobal.TriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')


process.simL1uGtDigis = cms.EDProducer("L1TGlobalProducer",
    #TechnicalTriggersUnprescaled = cms.bool(False),
    ProduceL1GtObjectMapRecord = cms.bool(True),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(1),
    L1DataBxInEvent = cms.int32(5),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    GmtInputTag = cms.InputTag(""),
    caloInputTag = cms.InputTag("caloStage2Digis"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    #WritePsbL1GtDaqRecord = cms.bool(True),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(0)
)

process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
                egInputTag    = cms.InputTag("caloStage2Digis"),
		muInputTag    = cms.InputTag(""),
		tauInputTag   = cms.InputTag("caloStage2Digis"),
		jetInputTag   = cms.InputTag("caloStage2Digis"),
		etsumInputTag = cms.InputTag("caloStage2Digis"),
		uGtRecInputTag = cms.InputTag("simL1uGtDigis"),
		uGtAlgInputTag = cms.InputTag("simL1uGtDigis"),
		uGtExtInputTag = cms.InputTag("simL1uGtDigis"),
		bxOffset       = cms.int32(skip),
		minBx          = cms.int32(0),
		maxBx          = cms.int32(0),
		minBxVec       = cms.int32(0),
		maxBxVec       = cms.int32(0),		
		dumpGTRecord   = cms.bool(True),
		dumpVectors    = cms.bool(True),
		tvFileName     = cms.string( ("TestVector_%03d.txt") % job )
		 )
		 
		 

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator
				     +process.esTest
				     +process.L1TCaloStage2
				    # +process.l1tStage2CaloAnalyzer
				    # +process.l1tStage2InputPatternWriter
                                     +process.simL1uGtDigis
                                     +process.dumpGTRecord
				     )
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.output_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.endjob_step,process.output_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 

