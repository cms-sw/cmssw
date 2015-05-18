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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(neventsPerJob)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
	"root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/06801062-95EA-E411-BAA4-002618943832.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/069BD085-9BEA-E411-B063-002618943985.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/0EABD22E-97EA-E411-9187-0025905A6068.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/10E5C59E-9DEA-E411-B378-00261894385D.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/14C238D9-96EA-E411-B343-00261894387E.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/2634383A-94EA-E411-B274-002590596486.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/36F53C7D-99EA-E411-995F-0026189437EB.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/3EC53EC3-A1EA-E411-A30F-002618943826.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/42B2E0FF-95EA-E411-9743-0025905B860C.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/444664B9-95EA-E411-844B-00261894391B.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/48D2AF08-98EA-E411-B556-002618943869.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/4A5FFFC2-95EA-E411-B62F-0025905B858C.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/5419CE7B-9AEA-E411-AE7C-002618943923.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/583FA9B9-98EA-E411-B867-0025905A7786.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/60CD47B5-95EA-E411-9469-00261894382D.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/647E2B80-A0EA-E411-80CB-002618943950.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/78A02D9A-94EA-E411-8625-0025905A6082.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/88637A5A-A0EA-E411-916B-0025905A6094.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/8E93F5C6-A0EA-E411-A4ED-0025905A6126.root",
        "root://cmsxrootd-site.fnal.gov//store/relval/CMSSW_7_5_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU50ns_MCRUN2_74_V6-v1/00000/908AF505-96EA-E411-9C1D-003048FFD756.root",
	),
    skipEvents = cms.untracked.uint32(skip)
    )

process.output =cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
	fileName = cms.untracked.string('testGlobalMCInputProducer_'+`job`+'.root')
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
		           muBx    = cms.untracked.vint32(0, -1,  0,  0,  1,  2),
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
			   jetHwEta = cms.untracked.vint32(  1,  19,  11,   0,  17,  11)
		       ),
		       
                       etsumParams = cms.untracked.PSet(
		           etsumBx    = cms.untracked.vint32( -2, -1,   0,  1,  2),
			   etsumHwPt  = cms.untracked.vint32(  2,  1, 204,  3,  4),  
			   etsumHwPhi = cms.untracked.vint32(  2,  1,  20,  3,  4)
		       )		       		       		       		       
                    )

## Load our L1 menu
process.load('L1Trigger.L1TGlobal.StableParametersConfig_cff')

process.load('L1Trigger.L1TGlobal.TriggerMenuXml_cfi')
process.TriggerMenuXml.TriggerMenuLuminosity = 'startup'
#process.TriggerMenuXml.DefXmlFile = 'L1_Example_Menu_2013.xml'
process.TriggerMenuXml.DefXmlFile = 'L1Menu_Reference_2014.xml'

process.load('L1Trigger.L1TGlobal.TriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')


process.simL1uGtDigis = cms.EDProducer("l1t::GtProducer",
    #TechnicalTriggersUnprescaled = cms.bool(False),
    ProduceL1GtObjectMapRecord = cms.bool(True),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(1),
    L1DataBxInEvent = cms.int32(5),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    GmtInputTag = cms.InputTag("gtInput"),
    caloInputTag = cms.InputTag("gtInput"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    #WritePsbL1GtDaqRecord = cms.bool(True),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(0)
)

process.dumpGTRecord = cms.EDAnalyzer("l1t::GtRecordDump",
                egInputTag    = cms.InputTag("gtInput"),
		muInputTag    = cms.InputTag("gtInput"),
		tauInputTag   = cms.InputTag("gtInput"),
		jetInputTag   = cms.InputTag("gtInput"),
		etsumInputTag = cms.InputTag("gtInput"),
		uGtRecInputTag = cms.InputTag("simL1uGtDigis"),
		uGtAlgInputTag = cms.InputTag("simL1uGtDigis"),
		uGtExtInputTag = cms.InputTag("simL1uGtDigis"),
		bxOffset       = cms.int32(skip),
		minBx          = cms.int32(-2),
		maxBx          = cms.int32(2),
		minBxVec       = cms.int32(0),
		maxBxVec       = cms.int32(0),		
		dumpGTRecord   = cms.bool(False),
		dumpVectors    = cms.bool(True),
		tvFileName     = cms.string( ("TestVector_%03d.txt") % job )
		 )



process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
process.l1GtTrigReport.L1GtRecordInputTag = "simL1uGtDigis"
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
    *process.simL1uGtDigis
    *process.dumpGTRecord
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

if dump:
    outfile = open('dump_runGlobalFakeInputProducer_'+`job`+'.py','w')
    print >> outfile,process.dumpPython()
    outfile.close()
