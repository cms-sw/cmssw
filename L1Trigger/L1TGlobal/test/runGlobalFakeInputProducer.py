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
newXML = False #whether running with the new Grammar

# Argument parsing
# vvv

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

neventsPerJob = nevents/njob
skip = job * neventsPerJob

if skip>4:
    skip = skip-4
    neventsPerJob = neventsPerJob+4

import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEMULATION')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')


# Select the Message Logger output you would like to see:
#
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('L1Trigger/L1TYellow/l1t_debug_messages_cfi')
#process.load('L1Trigger/L1TYellow/l1t_info_messages_cfi')

process.load('L1Trigger/L1TGlobal/debug_messages_cfi')
process.MessageLogger.l1t_debug.l1t.limit = cms.untracked.int32(100000)

process.MessageLogger.categories.append('l1t|Global')
# DEBUG
#process.MessageLogger.debugModules = cms.untracked.vstring('simGtStage2Digis') 
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG') 

# set the number of events
process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(10)
    input = cms.untracked.int32(neventsPerJob)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        "/store/mc/RunIISummer19UL18RECO/GluGluToContinToZZTo4mu_13TeV_MCFM701_pythia8/AODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/110000/664BBEBB-93A9-5B40-AD9C-DE835A79B712.root",
        #"/store/mc/RunIISummer20UL18RECO/BuToTau_To3Mu_MuFilter_TuneCP5_13TeV-pythia8-evtgen/AODSIM/106X_upgrade2018_realistic_v11_L1v1-v1/20000/00901296-D966-DF40-AA25-5F7A959B79CA.root",
        #"/store/mc/RunIISummer20UL18RECO/DsToTau_To3Mu_MuFilter_TuneCP5_13TeV-pythia8-evtgen/AODSIM/106X_upgrade2018_realistic_v11_L1v1-v1/00000/0003B7FD-6C1E-BF4C-8DA9-BA8A27AF0290.root",
        #"/store/mc/RunIISummer19UL18RECO/ZZ_TuneCP5_13TeV-pythia8/AODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/280000/04530FC4-E54D-D34A-950E-9F300321E037.root",
        #"/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/AODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/F03B8956-5D87-E511-8AE9-002590D0AFFC.root",
        #"/store/mc/RunIISummer19UL18HLT/TTTo2L2Nu_mtop178p5_TuneCP5_13TeV-powheg-pythia8/GEN-SIM-RAW/102X_upgrade2018_realistic_v15-v2/280000/00429618-85B5-124F-9C16-0C9F07A39E73.root+"
        #"/store/mc/PhaseIFall16DR/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/110000/444C2036-84FC-E611-A86D-02163E01433C.root",
        #"/store/mc/RunIISpring16DR80/TT_TuneCUETP8M1_13TeV-powheg-pythia8/GEN-SIM-RAW/FlatPU20to70HcalNZSRAW_withHLT_80X_mcRun2_asymptotic_v14_ext3-v1/50000/D6D4CAF2-AD65-E611-9642-001EC94BA169.root",
        #"/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root",
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


# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t_histos.root')

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
## process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, '106X_upgrade2018_realistic_v11', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '112X_mcRun2_asymptotic_v2', '')
## process.GlobalTag = GlobalTag(process.GlobalTag, '90X_upgrade2017_realistic_PerfectEcalIc_EGM_PFCalib', '')
## auto:upgradePLS1
## 81X_upgrade2017_realistic_v26
## 80X_mcRun2_asymptotic_v14

## ## needed until prescales go into GlobalTag ########################
## from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
## process.l1conddb = cms.ESSource("PoolDBESSource",
##        CondDBSetup,
##        connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS'),
##        toGet   = cms.VPSet(
##             cms.PSet(
##                  record = cms.string('L1TGlobalPrescalesVetosRcd'),
##                  tag = cms.string("L1TGlobalPrescalesVetos_passThrough_mc")
##             )
##        )
## )
## process.es_prefer_l1conddb = cms.ESPrefer( "PoolDBESSource","l1conddb")
## # done ##############################################################

# Flag to switch between using MC particles and injecting individual particles
useMCtoGT = True

process.dumpGT = cms.EDAnalyzer("l1t::GtInputDump",
                egInputTag    = cms.InputTag("gtInput"),
		muInputTag    = cms.InputTag("gtInput"),
		tauInputTag   = cms.InputTag("gtInput"),
		jetInputTag   = cms.InputTag("gtInput"),
		etsumInputTag = cms.InputTag("gtInput"),
		minBx         = cms.int32(0),
		maxBx         = cms.int32(0)
		 )
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.mcL1GTinput = cms.EDProducer("l1t::GenToInputProducer",
                                     bxFirst = cms.int32(-2),
                                     bxLast = cms.int32(2),
				     maxMuCand = cms.int32(8),
				     maxJetCand = cms.int32(12),
				     maxEGCand  = cms.int32(12),
				     maxTauCand = cms.int32(8),
                                     jetEtThreshold = cms.double(1),
                                     tauEtThreshold = cms.double(1),
                                     egEtThreshold  = cms.double(1),
                                     muEtThreshold  = cms.double(1),
				     emptyBxTrailer = cms.int32(5),
				     emptyBxEvt = cms.int32(neventsPerJob)
                                     )

process.mcL1GTinput.maxMuCand = cms.int32(8)
process.mcL1GTinput.maxJetCand = cms.int32(12)
process.mcL1GTinput.maxEGCand  = cms.int32(12)
process.mcL1GTinput.maxTauCand = cms.int32(8)

# Fake the input
process.fakeL1GTinput = cms.EDProducer("l1t::FakeInputProducer",

# Note: There is no error checking on these parameters...you are responsible.
                       egParams = cms.untracked.PSet(
		           egBx    = cms.untracked.vint32(-2, -1,  0,  0,  1,  2),
			   egHwPt  = cms.untracked.vint32(10, 20, 30, 61, 40, 50),
			   egHwPhi = cms.untracked.vint32(11, 21, 31, 61, 41, 51),
			   egHwEta = cms.untracked.vint32(12, 22, 32, 62, 42, 52),
			   egIso   = cms.untracked.vint32( 0,  0,  1,  1,  0,  0)
		       ),

                       muParams = cms.untracked.PSet(
		           muBx    = cms.untracked.vint32(-2, -1,  0,  0,  1,  2),
			   muHwPt  = cms.untracked.vint32(5, 20, 30, 61, 40, 50),
			   muHwPhi = cms.untracked.vint32(11, 21, 31, 61, 41, 51),
			   muHwEta = cms.untracked.vint32(12, 22, 32, 62, 42, 52),
			   muIso   = cms.untracked.vint32( 0,  0,  1,  1,  0,  0)
		       ),

                       tauParams = cms.untracked.PSet(
		           tauBx    = cms.untracked.vint32(),
			   tauHwPt  = cms.untracked.vint32(),
			   tauHwPhi = cms.untracked.vint32(),
			   tauHwEta = cms.untracked.vint32(),
			   tauIso   = cms.untracked.vint32()
		       ),

                       jetParams = cms.untracked.PSet(
		           jetBx    = cms.untracked.vint32(  0,   0,   2,   1,   1,   2),
			   jetHwPt  = cms.untracked.vint32(100, 200, 130, 170,  85, 145),
			   jetHwPhi = cms.untracked.vint32(  2,  67,  10,   3,  78,  10),
			   jetHwEta = cms.untracked.vint32(  110,  -99,  11,   0,  17,  11)
		       ),

                       etsumParams = cms.untracked.PSet(
		           etsumBx    = cms.untracked.vint32( -2, -1,   0,  1,  2),
			   etsumHwPt  = cms.untracked.vint32(  2,  1, 204,  3,  4),
			   etsumHwPhi = cms.untracked.vint32(  2,  1,  20,  3,  4)
		       )
                    )

## Load our L1 menu
process.load('L1Trigger.L1TGlobal.GlobalParameters_cff')

process.load("L1Trigger.L1TGlobal.TriggerMenu_cff")

xmlMenu="L1Menu_test_mass_3_body_reduced_v2.xml"
process.TriggerMenu.L1TriggerMenuFile = cms.string(xmlMenu)
process.ESPreferL1TXML = cms.ESPrefer("L1TUtmTriggerMenuESProducer","TriggerMenu")

process.dumpMenu = cms.EDAnalyzer("L1MenuViewer")
# DEBUG: Information about names and types of algos parsed by the emulator from the menu
#process.menuDumper = cms.EDAnalyzer("L1TUtmTriggerMenuDumper") 

## Fill External conditions
process.load('L1Trigger.L1TGlobal.simGtExtFakeProd_cfi')
process.simGtExtFakeProd.bxFirst = cms.int32(-2)
process.simGtExtFakeProd.bxLast = cms.int32(2)
process.simGtExtFakeProd.setBptxAND   = cms.bool(True)
process.simGtExtFakeProd.setBptxPlus  = cms.bool(True)
process.simGtExtFakeProd.setBptxMinus = cms.bool(True)
process.simGtExtFakeProd.setBptxOR    = cms.bool(True)


## Run the Stage 2 uGT emulator
process.load('L1Trigger.L1TGlobal.simGtStage2Digis_cfi')
process.simGtStage2Digis.PrescaleCSVFile = cms.string('prescale_L1TGlobal.csv')
process.simGtStage2Digis.PrescaleSet = cms.uint32(1)
process.simGtStage2Digis.ExtInputTag = cms.InputTag("simGtExtFakeProd")
process.simGtStage2Digis.MuonInputTag = cms.InputTag("gtInput")
process.simGtStage2Digis.EGammaInputTag = cms.InputTag("gtInput")
process.simGtStage2Digis.TauInputTag = cms.InputTag("gtInput")
process.simGtStage2Digis.JetInputTag = cms.InputTag("gtInput")
process.simGtStage2Digis.EtSumInputTag = cms.InputTag("gtInput")
process.simGtStage2Digis.EmulateBxInEvent = cms.int32(1)
#process.simGtStage2Digis.Verbosity = cms.untracked.int32(1)
#process.simGtStage2Digis.AlgorithmTriggersUnprescaled = cms.bool(True)
#process.simGtStage2Digis.AlgorithmTriggersUnmasked = cms.bool(True)

process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
                egInputTag    = cms.InputTag("gtInput"),
		muInputTag    = cms.InputTag("gtInput"),
		tauInputTag   = cms.InputTag("gtInput"),
		jetInputTag   = cms.InputTag("gtInput"),
		etsumInputTag = cms.InputTag("gtInput"),
		uGtAlgInputTag = cms.InputTag("simGtStage2Digis"),
		uGtExtInputTag = cms.InputTag("simGtExtFakeProd"),
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
process.l1GtTrigReport.PrintVerbosity = 2
process.report = cms.Path(process.l1GtTrigReport)

process.MessageLogger.categories.append("MuConditon")

if useMCtoGT:
    process.gtInput = process.mcL1GTinput.clone()
else:
    process.gtInput = process.fakeL1GTinput.clone()

# Setup Digi to Raw to Digi
process.load('EventFilter.L1TRawToDigi.gtStage2Raw_cfi')
process.gtStage2Raw.GtInputTag = cms.InputTag("simGtStage2Digis")
process.gtStage2Raw.ExtInputTag = cms.InputTag("simGtExtFakeProd")
process.gtStage2Raw.EGammaInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.TauInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.JetInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.EtSumInputTag = cms.InputTag("gtInput")
process.gtStage2Raw.MuonInputTag = cms.InputTag("gtInput")

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

# gt analyzer
process.l1tGlobalAnalyzer = cms.EDAnalyzer('L1TGlobalAnalyzer',
    doText = cms.untracked.bool(False),
    gmuToken = cms.InputTag("None"),
    dmxEGToken = cms.InputTag("None"),
    dmxTauToken = cms.InputTag("None"),
    dmxJetToken = cms.InputTag("None"),
    dmxEtSumToken = cms.InputTag("None"),
    muToken = cms.InputTag("gtInput"),
    egToken = cms.InputTag("gtInput"),
    tauToken = cms.InputTag("gtInput"),
    jetToken = cms.InputTag("gtInput"),
    etSumToken = cms.InputTag("gtInput"),
    gtAlgToken = cms.InputTag("simGtStage2Digis"),
    emulDxAlgToken = cms.InputTag("None"),
    emulGtAlgToken = cms.InputTag("simGtStage2Digis")
)



process.p1 = cms.Path(

## Generate input, emulate, dump results
    process.dumpMenu
    *process.gtInput
#    *process.dumpGT
    *process.simGtExtFakeProd
    *process.simGtStage2Digis
    *process.dumpGTRecord

## Sequence for packing and unpacking uGT data
#    +process.gtStage2Raw
#    +process.dumpRaw
#    +process.newGtStage2Digis
#    +process.newDumpGTRecord

## Analysis/Dumping
    *process.l1tGlobalAnalyzer
#    *process.menuDumper # DEBUG -> to activate the menuDumper
#    *process.debug
#    *process.dumpED
#    *process.dumpES
    )

process.schedule = cms.Schedule(
    process.p1
    )
#process.schedule.append(process.report)
if rootout:
    process.outpath = cms.EndPath(process.output)
    process.schedule.append(process.outpath)

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

# Options for multithreading
#process.options.numberOfThreads = cms.untracked.uint32( 2 )
#process.options.numberOfStreams = cms.untracked.uint32( 0 )

if dump:
    outfile = open('dump_runGlobalFakeInputProducer_'+repr(job)+'.py','w')
    print(process.dumpPython(), file=outfile)
    outfile.close()
