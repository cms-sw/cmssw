
import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEMULATION')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')



# Select the Message Logger output you would like to see:
#
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('L1Trigger/L1TYellow/l1t_debug_messages_cfi')
#process.load('L1Trigger/L1TYellow/l1t_info_messages_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    #fileNames = cms.untracked.vstring("/store/RelVal/CMSSW_7_0_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_0_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/22610530-FC24-E311-AF35-003048FFD7C2.root")
    fileNames = cms.untracked.vstring("file:/home/puigh/work/L1Upgrade/CMSSW_6_2_0/src/Neutrino_Pt2to20_gun_UpgradeL1TDR-PU50_POSTLS161_V12-v1_001D5CFF-2839-E211-9777-0030487FA483.root")
    )

process.output =cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
	fileName = cms.untracked.string('testFakeProducer.root')
	)
	
process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')

process.dumpGT = cms.EDAnalyzer("l1t::L1TGlobalInputTester",
                egInputTag    = cms.InputTag("fakeL1TGinput"),
		muInputTag    = cms.InputTag("fakeL1TGinput"),
		tauInputTag   = cms.InputTag("fakeL1TGinput"),
		jetInputTag   = cms.InputTag("fakeL1TGinput"),
		etsumInputTag = cms.InputTag("fakeL1TGinput") 
		 )
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")


# Fake the input
process.fakeL1TGinput  = cms.EDProducer("l1t::L1TGlobalFakeInputProducer",

# Note: There is no error checking on these parameters...you are responsible. 
                       egParams = cms.untracked.PSet(
		           egBx    = cms.untracked.vint32(-2, -1,  0,  0,  1,  2),
			   egHwPt  = cms.untracked.vint32(10, 20, 30, 61, 40, 50),
			   egHwPhi = cms.untracked.vint32(11, 21, 31, 61, 41, 51),
			   egHwEta = cms.untracked.vint32(12, 22, 32, 62, 42, 52),
			   egIso   = cms.untracked.vint32( 0,  0,  1,  1,  0,  0)
		       ),
		       
                       muParams = cms.untracked.PSet(
		           muBx    = cms.untracked.vint32(),
			   muHwPt  = cms.untracked.vint32(),
			   muHwPhi = cms.untracked.vint32(),
			   muHwEta = cms.untracked.vint32(),
			   muIso   = cms.untracked.vint32()
		       ),

                       tauParams = cms.untracked.PSet(
		           tauBx    = cms.untracked.vint32(),
			   tauHwPt  = cms.untracked.vint32(),
			   tauHwPhi = cms.untracked.vint32(),
			   tauHwEta = cms.untracked.vint32(),
			   tauIso   = cms.untracked.vint32()
		       ),
		       
                       jetParams = cms.untracked.PSet(
		           jetBx    = cms.untracked.vint32(  0,   0),
			   jetHwPt  = cms.untracked.vint32(100, 200),
			   jetHwPhi = cms.untracked.vint32( 10,  20),
			   jetHwEta = cms.untracked.vint32( 11,  21)
		       ),
		       
                       etsumParams = cms.untracked.PSet(
		           etsumBx    = cms.untracked.vint32( -2, -1,   0,  1,  2),
			   etsumHwPt  = cms.untracked.vint32(  2,  1, 204,  3,  4),  
			   etsumHwPhi = cms.untracked.vint32(  2,  1,  20,  3,  4)
		       )		       		       		       		       
                    )


process.p1 = cms.Path(
    process.fakeL1TGinput
    *process.dumpGT
#    * process.debug
#    *process.dumpED
#    *process.dumpES
    )

process.schedule = cms.Schedule(
    process.p1
    )
#process.outpath = cms.EndPath(process.output)
#process.schedule.append(process.outpath)

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
