
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

process.load('L1Trigger/L1TGlobal/l1tGt_debug_messages_cfi')
process.MessageLogger.l1t_debug.l1t.limit = cms.untracked.int32(100000)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    ### Neutrino Gun Sample - PU50
    #fileNames = cms.untracked.vstring("file:/home/puigh/work/L1Upgrade/CMSSW_6_2_0/src/Neutrino_Pt2to20_gun_UpgradeL1TDR-PU50_POSTLS161_V12-v1_001D5CFF-2839-E211-9777-0030487FA483.root")
    ### RelValTTBar
    #fileNames = cms.untracked.vstring("root://xrootd.unl.edu//store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2_amend-v4/00000/3A11157B-ED51-E311-BA75-003048679080.root")
    ### Local RelValTTBar
    fileNames = cms.untracked.vstring("/store/user/puigh/RelValTTbar_GEN-SIM-DIGI-RAW-HLTDEBUG_START70_V2_amend-v4_00000_3A11157B-ED51-E311-BA75-003048679080.root")
    ### RelValSingleElectronPt10
    #fileNames = cms.untracked.vstring("root://xrootd.unl.edu//store/relval/CMSSW_7_0_0_pre8/RelValSingleElectronPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2_amend-v4/00000/52DE2A7D-E651-E311-8E12-003048FFCBFC.root")
    )

process.output =cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
	fileName = cms.untracked.string('testGlobalMCInputProducer.root')
	)
	
process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')

# Flag to switch between using MC particles and injecting individual particles
useMCtoGT = True

process.dumpGT = cms.EDAnalyzer("l1t::L1TGlobalInputTester",
                egInputTag    = cms.InputTag("gtInput"),
		muInputTag    = cms.InputTag("gtInput"),
		tauInputTag   = cms.InputTag("gtInput"),
		jetInputTag   = cms.InputTag("gtInput"),
		etsumInputTag = cms.InputTag("gtInput") 
		 )
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.mcL1GTinput = cms.EDProducer("l1t::L1uGtGenToInputProducer",
                                     bxFirst = cms.int32(-2),
                                     bxLast = cms.int32(2),
                                     jetEtThreshold = cms.double(1),
                                     tauEtThreshold = cms.double(1),
                                     egEtThreshold  = cms.double(1),
                                     muEtThreshold  = cms.double(1)
                                     )
# Fake the input
process.fakeL1GTinput = cms.EDProducer("l1t::L1TGlobalFakeInputProducer",

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
		           jetBx    = cms.untracked.vint32(  0,   0,  2, -1, 2),
			   jetHwPt  = cms.untracked.vint32(100, 200,130,170,145),
			   jetHwPhi = cms.untracked.vint32( 10,  10, 10, 10, 10),
			   jetHwEta = cms.untracked.vint32( 11,  11, 11, 11, 11)
		       ),
		       
                       etsumParams = cms.untracked.PSet(
		           etsumBx    = cms.untracked.vint32( -2, -1,   0,  1,  2),
			   etsumHwPt  = cms.untracked.vint32(  2,  1, 204,  3,  4),  
			   etsumHwPhi = cms.untracked.vint32(  2,  1,  20,  3,  4)
		       )		       		       		       		       
                    )

## Load our L1 menu
process.load('L1Trigger.L1TGlobal.l1uGtTriggerMenuXml_cfi')
process.l1uGtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
#process.l1uGtTriggerMenuXml.DefXmlFile = 'L1_Example_Menu_2013.xml'
process.l1uGtTriggerMenuXml.DefXmlFile = 'L1Menu_Reference_2014.xml'

process.load('L1Trigger.L1TGlobal.L1uGtTriggerMenuConfig_cff')
process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::L1uGtTriggerMenuXmlProducer','l1uGtTriggerMenuXml')


process.simL1uGtDigis = cms.EDProducer("l1t::L1uGtProducer",
    #TechnicalTriggersUnprescaled = cms.bool(False),
    ProduceL1GtObjectMapRecord = cms.bool(True),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(5),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    #ReadTechnicalTriggerRecords = cms.bool(True),
    RecordLength = cms.vint32(3, 0),
    #TechnicalTriggersUnmasked = cms.bool(False),
    #ProduceL1GtEvmRecord = cms.bool(True),
    #GmtInputTag = cms.InputTag("gtDigis"),
    GmtInputTag = cms.InputTag("gtInput"),
    #TechnicalTriggersVetoUnmasked = cms.bool(False),
    #AlternativeNrBxBoardEvm = cms.uint32(0),
    #TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag("simBscDigis"), cms.InputTag("simRpcTechTrigDigis"), cms.InputTag("simHcalTechTrigDigis")),
    #CastorInputTag = cms.InputTag("castorL1Digis"),
    #GctInputTag = cms.InputTag("gctReEmulDigis"),
    caloInputTag = cms.InputTag("gtInput"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    #WritePsbL1GtDaqRecord = cms.bool(True),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(1)
)

process.dumpGTRecord = cms.EDAnalyzer("l1t::L1uGtRecordDump",
                egInputTag    = cms.InputTag("gtInput"),
		muInputTag    = cms.InputTag("gtInput"),
		tauInputTag   = cms.InputTag("gtInput"),
		jetInputTag   = cms.InputTag("gtInput"),
		etsumInputTag = cms.InputTag("gtInput"),
		uGtRecInputTag = cms.InputTag("simL1uGtDigis"),
		uGtAlgInputTag = cms.InputTag("simL1uGtDigis"),
		uGtExtInputTag = cms.InputTag("simL1uGtDigis") 
		 )



process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
process.l1GtTrigReport.L1GtRecordInputTag = "simL1uGtDigis"
process.l1GtTrigReport.PrintVerbosity = 2
process.report = cms.Path(process.l1GtTrigReport)

process.MessageLogger.categories.append("L1uGtMuonConditon")

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
process.outpath = cms.EndPath(process.output)
process.schedule.append(process.outpath)

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

outfile = open('dump_runGlobalFakeInputProducer.py','w')
print >> outfile,process.dumpPython()
outfile.close()
