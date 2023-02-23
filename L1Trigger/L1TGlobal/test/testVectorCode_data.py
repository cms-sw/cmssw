#!/usr/bin/env python
from __future__ import print_function
import sys

"""
The parameters can be changed by adding command line arguments of the form:
    testVectorCode_data.py nevents=-1
The latter can be used to change parameters in crab.
"""

job = 0 #job number
njob = 1 #number of jobs
nevents = 3564 #number of events
rootout = False #whether to produce root file
dump = False #dump python
newXML = False #whether running with the new Grammar

# ----------------
# Argument parsing
# ----------------
if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
    sys.argv.pop(0)
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

neventsPerJob = int(nevents/njob)
skip = job * neventsPerJob

if skip>4:
    skip = skip-4
    neventsPerJob = neventsPerJob+4

# ------------------------------------------------------------
# Set up Run 3 conditions to get the proper emulation sequence
# ------------------------------------------------------------
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('L1TEMULATION', Run3)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

# ---------------------
# Message Logger output
# ---------------------
process.load('FWCore.MessageService.MessageLogger_cfi')

# DEBUG
process.load('L1Trigger/L1TGlobal/debug_messages_cfi')
process.MessageLogger.l1t_debug.l1t.limit = cms.untracked.int32(100000)
process.MessageLogger.categories.append('l1t|Global')
# DEBUG
#process.MessageLogger.debugModules = cms.untracked.vstring('simGtStage2Digis') 
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG') 

# ------------
# Input source
# ------------
# Set the number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(neventsPerJob)
    )

# Set file: it needs to be a RAW format
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        "/store/data/Run2022G/EphemeralHLTPhysics0/RAW/v1/000/362/720/00000/36f350d4-8e8a-4e38-b399-77ad9bf351dc.root"
	),
    skipEvents = cms.untracked.uint32(skip)
    )

process.output =cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
	fileName = cms.untracked.string('testGlobalMCInputProducer_'+repr(job)+'.root')
	)

process.options = cms.untracked.PSet(
    wantSummary = cms.bool(True)
)

# -----------------------------------------------
# Additional output definition: TTree output file
# -----------------------------------------------
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t_histos.root')

# ----------
# Global Tag
# ----------
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '124X_dataRun3_Prompt_v4', '')

# ----------------
# Load the L1 menu
# ----------------
process.load('L1Trigger.L1TGlobal.GlobalParameters_cff')
process.load("L1Trigger.L1TGlobal.TriggerMenu_cff")
xmlMenu="L1Menu_Collisions2022_v1_4_0.xml"
process.TriggerMenu.L1TriggerMenuFile = cms.string(xmlMenu)
process.ESPreferL1TXML = cms.ESPrefer("L1TUtmTriggerMenuESProducer","TriggerMenu")

process.dumpMenu = cms.EDAnalyzer("L1MenuViewer")
# DEBUG: Information about names and types of algos parsed by the emulator from the menu
#process.menuDumper = cms.EDAnalyzer("L1TUtmTriggerMenuDumper") 

# -----------------------------------------
# Load the GT inputs from the unpacker step
# -----------------------------------------
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.raw2digi_step = cms.Path(process.RawToDigi)

process.dumpGT = cms.EDAnalyzer("l1t::GtInputDump",
                egInputTag       = cms.InputTag("gtInput"),
		muInputTag       = cms.InputTag("gtInput"),
		muShowerInputTag = cms.InputTag("gtInput"),
		tauInputTag      = cms.InputTag("gtInput"),
		jetInputTag      = cms.InputTag("gtInput"),
		etsumInputTag    = cms.InputTag("gtInput"),
		minBx            = cms.int32(0),
		maxBx            = cms.int32(0)
		 )
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

# ------------------------
# Fill External conditions
# ------------------------
process.load('L1Trigger.L1TGlobal.simGtExtFakeProd_cfi')
process.simGtExtFakeProd.bxFirst = cms.int32(-2)
process.simGtExtFakeProd.bxLast = cms.int32(2)
process.simGtExtFakeProd.setBptxAND   = cms.bool(True)
process.simGtExtFakeProd.setBptxPlus  = cms.bool(True)
process.simGtExtFakeProd.setBptxMinus = cms.bool(True)
process.simGtExtFakeProd.setBptxOR    = cms.bool(True)

# ----------------------------
# Run the Stage 2 uGT emulator
# ----------------------------
process.load('L1Trigger.L1TGlobal.simGtStage2Digis_cfi')
process.simGtStage2Digis.PrescaleSet = cms.uint32(1)
process.simGtStage2Digis.ExtInputTag = cms.InputTag("simGtExtFakeProd")
process.simGtStage2Digis.MuonInputTag = cms.InputTag("gtStage2Digis", "Muon")
process.simGtStage2Digis.MuonShowerInputTag = cms.InputTag("gtStage2Digis", "MuonShower")
process.simGtStage2Digis.EGammaInputTag = cms.InputTag("gtStage2Digis", "EGamma")
process.simGtStage2Digis.TauInputTag = cms.InputTag("gtStage2Digis", "Tau")
process.simGtStage2Digis.JetInputTag = cms.InputTag("gtStage2Digis", "Jet")
process.simGtStage2Digis.EtSumInputTag = cms.InputTag("gtStage2Digis", "ETSum")
process.simGtStage2Digis.EmulateBxInEvent = cms.int32(1)

process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
                                      egInputTag       = cms.InputTag("gtStage2Digis", "EGamma"),
		                      muInputTag       = cms.InputTag("gtStage2Digis", "Muon"),
		                      muShowerInputTag = cms.InputTag("gtStage2Digis", "MuonShower"),
		                      tauInputTag      = cms.InputTag("gtStage2Digis", "Tau"),
                                      jetInputTag      = cms.InputTag("gtStage2Digis", "Jet"),
                                      etsumInputTag    = cms.InputTag("gtStage2Digis", "ETSum"),
                                      uGtAlgInputTag   = cms.InputTag("simGtStage2Digis"),
                                      uGtExtInputTag   = cms.InputTag("simGtExtFakeProd"),
                                      uGtObjectMapInputTag = cms.InputTag("simGtStage2Digis"),
                                      bxOffset       = cms.int32(skip),
                                      minBx          = cms.int32(-2),
                                      maxBx          = cms.int32(2),
                                      minBxVec       = cms.int32(0),
                                      maxBxVec       = cms.int32(0),
                                      dumpGTRecord   = cms.bool(True),
                                      dumpGTObjectMap= cms.bool(False),
                                      dumpTrigResults= cms.bool(False),
                                      dumpVectors    = cms.bool(True),
                                      tvFileName     = cms.string( ("TestVector_%03d.txt") % job ),
                                      tvVersion      = cms.int32(3),
                                      ReadPrescalesFromFile = cms.bool(True),
                                      psFileName     = cms.string( "prescale_L1TGlobal.csv" ),
                                      psColumn       = cms.int32(1),
                                      unprescaleL1Algos = cms.bool(False),
                                      unmaskL1Algos     = cms.bool(False)
		 )

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
process.l1GtTrigReport.L1GtRecordInputTag = "simGtStage2Digis"
process.l1GtTrigReport.PrintVerbosity = 0 
process.report = cms.Path(process.l1GtTrigReport)

process.MessageLogger.categories.append("MuConditon")

# -------------------------
# Setup Digi to Raw to Digi
# -------------------------
process.load('EventFilter.L1TRawToDigi.gtStage2Raw_cfi')
process.gtStage2Raw.GtInputTag = cms.InputTag("simGtStage2Digis")
process.gtStage2Raw.ExtInputTag = cms.InputTag("simGtExtFakeProd")
process.gtStage2Raw.EGammaInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.TauInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.JetInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.EtSumInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.MuonInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.MuonShowerInputTag = cms.InputTag("gtInput")

process.load('EventFilter.L1TRawToDigi.gtStage2Digis_cfi')
process.newGtStage2Digis = process.gtStage2Digis.clone()
process.newGtStage2Digis.InputLabel = cms.InputTag('gtStage2Raw')
# DEBUG 
#process.newGtStage2Digis.debug = cms.untracked.bool(True) 

process.dumpRaw = cms.EDAnalyzer(
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("gtStage2Raw"),
    feds = cms.untracked.vint32 ( 1404 ),
    dumpPayload = cms.untracked.bool ( True )
)

process.newDumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
                egInputTag    = cms.InputTag("newGtStage2Digis","EGamma"),
		muInputTag    = cms.InputTag("newGtStage2Digis","Muon"),
		muShowerInputTag    = cms.InputTag("newGtStage2Digis","MuonShower"),
		tauInputTag   = cms.InputTag("newGtStage2Digis","Tau"),
		jetInputTag   = cms.InputTag("newGtStage2Digis","Jet"),
		etsumInputTag = cms.InputTag("newGtStage2Digis","EtSum"),
		uGtAlgInputTag = cms.InputTag("newGtStage2Digis"),
		uGtExtInputTag = cms.InputTag("newGtStage2Digis"),
		uGtObjectMapInputTag = cms.InputTag("simGtStage2Digis"),
		bxOffset       = cms.int32(skip),
		minBx          = cms.int32(0),
		maxBx          = cms.int32(0),
		minBxVec       = cms.int32(0),
		maxBxVec       = cms.int32(0),
		dumpGTRecord   = cms.bool(True),
		dumpGTObjectMap= cms.bool(True),
                dumpTrigResults= cms.bool(False),
		dumpVectors    = cms.bool(False),
		tvFileName     = cms.string( ("TestVector_%03d.txt") % job ),
                ReadPrescalesFromFile = cms.bool(False),
                psFileName     = cms.string( "prescale_L1TGlobal.csv" ),
                psColumn       = cms.int32(1)
		 )

# -----------
# GT analyzer
# -----------
process.l1tGlobalAnalyzer = cms.EDAnalyzer('L1TGlobalAnalyzer',
                                           doText = cms.untracked.bool(False),
                                           gmuToken = cms.InputTag("None"),
                                           dmxEGToken = cms.InputTag("None"),
                                           dmxTauToken = cms.InputTag("None"),
                                           dmxJetToken = cms.InputTag("None"),
                                           dmxEtSumToken = cms.InputTag("None"),
                                           muToken = cms.InputTag("gtStage2Digis", "Muon"),
                                           muShowerToken = cms.InputTag("gtStage2Digis", "MuonShower"),
                                           egToken = cms.InputTag("gtStage2Digis", "EGamma"),
                                           tauToken = cms.InputTag("gtStage2Digis", "Tau"),
                                           jetToken = cms.InputTag("gtStage2Digis", "Jet"),
                                           etSumToken = cms.InputTag("gtStage2Digis", "EtSum"),
                                           gtAlgToken = cms.InputTag("simGtStage2Digis"),
                                           emulDxAlgToken = cms.InputTag("None"),
                                           emulGtAlgToken = cms.InputTag("simGtStage2Digis")
)

# ------------------
# Process definition
# ------------------
process.p1 = cms.Path(
    ## Input, emulation, dump of the results
    process.dumpMenu
    *process.RawToDigi 
    #*process.gtInput
    #*process.dumpGT
    *process.simGtExtFakeProd
    *process.simGtStage2Digis
    *process.dumpGTRecord

    ## Sequence for packing and unpacking uGT data
    #+process.gtStage2Raw
    #+process.dumpRaw
    #+process.newGtStage2Digis
    #+process.newDumpGTRecord

    ## Analysis/Dumping
    *process.l1tGlobalAnalyzer
    #*process.menuDumper # DEBUG -> to activate the menuDumper
    #*process.debug
    #*process.dumpED
    #*process.dumpES
    )

# -------------------
# Schedule definition
# -------------------
process.schedule = cms.Schedule(
    process.p1
)
#process.schedule.append(process.report)

if rootout:
    process.outpath = cms.EndPath(process.output)
    process.schedule.append(process.outpath)

# Spit out filter efficiency at the end
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

# Options for multithreading
#process.options.numberOfThreads = cms.untracked.uint32( 2 )
#process.options.numberOfStreams = cms.untracked.uint32( 0 )

if dump:
    outfile = open('dump_runGlobalFakeInputProducer_'+repr(job)+'.py','w')
    print(process.dumpPython(), file=outfile)
    outfile.close()
