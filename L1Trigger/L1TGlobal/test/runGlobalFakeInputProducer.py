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
    if type(globals()[k]) == bool:
        globals()[k] = v.lower() in ('y', 'yes', 'true', 't', '1')
    elif type(globals()[k]) == int:
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

#process.MessageLogger.categories.append('l1t|Global')
#process.MessageLogger.debugModules = cms.untracked.vstring('*')
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(neventsPerJob)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        #"/store/user/puigh/L1Upgrade/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_7_6_0/4C462F65-9F7F-E511-972A-0026189438A9.root",
        "/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root",
        #"root://xrootd.ba.infn.it//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/4C462F65-9F7F-E511-972A-0026189438A9.root",
        #"root://xrootd.ba.infn.it//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/703E7EAB-9D7F-E511-B886-003048FFCBFC.root",
        #"root://xrootd.ba.infn.it//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/8AF07AAB-9D7F-E511-B8B4-003048FFCBFC.root",
        #"root://xrootd.ba.infn.it//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/962BEF7C-9D7F-E511-A2BB-0025905B85AA.root",
        #"root://xrootd.ba.infn.it//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/C409A519-9E7F-E511-BD4C-0025905B8590.root",
        #"root://xrootd.ba.infn.it//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/E8D41D6A-9F7F-E511-A10A-003048FFD740.root",
        #"root://xrootd.ba.infn.it//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/EE048767-9E7F-E511-B1AA-0025905B8606.root",
        #"root://xrootd.ba.infn.it//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/4431031E-9E7F-E511-9F42-0025905938A4.root",
        #"root://cmsxrootd.fnal.gov//store/relval/CMSSW_7_6_0/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v11-v1/00000/4431031E-9E7F-E511-9F42-0025905938A4.root",
	),
    skipEvents = cms.untracked.uint32(skip)
    )

process.output =cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
	fileName = cms.untracked.string('testGlobalMCInputProducer_'+repr(job)+'.root')
	)
	
process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')

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
process.load('L1Trigger.L1TGlobal.StableParameters_cff')

process.load("L1Trigger.L1TGlobal.TriggerMenu_cff")
process.TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_Collisions2016_dev_v3.xml')
#process.TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_Collisions2015_25nsStage1_v7_uGT.xml')

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
#process.simGtStage2Digis.Verbosity = cms.untracked.int32(1)


process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
                egInputTag    = cms.InputTag("gtInput"),
		muInputTag    = cms.InputTag("gtInput"),
		tauInputTag   = cms.InputTag("gtInput"),
		jetInputTag   = cms.InputTag("gtInput"),
		etsumInputTag = cms.InputTag("gtInput"),
		uGtAlgInputTag = cms.InputTag("simGtStage2Digis"),
		uGtExtInputTag = cms.InputTag("simGtExtFakeProd"),
		bxOffset       = cms.int32(skip),
		minBx          = cms.int32(0),
		maxBx          = cms.int32(0),
		minBxVec       = cms.int32(0),
		maxBxVec       = cms.int32(0),		
		dumpGTRecord   = cms.bool(True),
                dumpTrigResults= cms.bool(True),
		dumpVectors    = cms.bool(True),
		tvFileName     = cms.string( ("TestVector_%03d.txt") % job ),
                psFileName     = cms.string( "prescale_L1TGlobal.csv" ),
                psColumn       = cms.int32(1)
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




process.p1 = cms.Path(
    process.gtInput
#    *process.dumpGT
    *process.simGtExtFakeProd
    *process.simGtStage2Digis
    *process.dumpGTRecord
#    +process.menuDumper
#    * process.debug
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
    print >> outfile,process.dumpPython()
    outfile.close()
