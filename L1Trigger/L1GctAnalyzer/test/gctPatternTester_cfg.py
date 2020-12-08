#
# This configuration is intended to be used to check pattern tests with the GCT emulator 
#
# Alex Tapper 8/9/10
#

process = cms.Process('GctPatternTester')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source ( "EmptySource" )

# One orbit of data is default for capture 
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 3563 ) )

# Input captured ascii file
process.gctRaw = cms.EDProducer( "TextToRaw",
                                   filename = cms.untracked.string ( "patternCapture_ts__2010_09_03__13h19m20s.txt" ),
                                   GctFedId = cms.untracked.int32 ( 745 )
                                   )

# Settings for pattern test (corresponds to V38_FS_Int11_Tau2_AllPipes_VME key)
process.load('L1Trigger.L1GctAnalyzer.gctPatternTestConfig_cff')

# GCT emulator
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
process.valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
process.valGctDigis.inputLabel = cms.InputTag("l1GctHwDigis")
process.valGctDigis.writeInternalData = cms.bool(True)
process.valGctDigis.preSamples = cms.uint32(0)
process.valGctDigis.postSamples = cms.uint32(0)
process.valGctDigis.useImprovedTauAlgorithm = cms.bool(True)

# GCT unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.inputLabel = cms.InputTag( "gctRaw" )

# L1Comparator
process.load('L1Trigger.HardwareValidation.L1Comparator_cfi')
process.l1compare.GCTsourceData = cms.InputTag("l1GctHwDigis")
process.l1compare.COMPARE_COLLS = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
process.l1compare.DumpMode = cms.untracked.int32(1)
process.l1compare.DumpFile = cms.untracked.string('dump.txt')
process.l1compare.VerboseFlag = cms.untracked.int32(0)

# Dump GCT digis
process.load('L1Trigger.L1GctAnalyzer.dumpGctDigis_cfi')
process.dumpGctDigis.doRctEm = cms.untracked.bool(False)
process.dumpGctDigis.doEm = cms.untracked.bool(True)
process.dumpGctDigis.doJets = cms.untracked.bool(True)
process.dumpGctDigis.doEmulated = cms.untracked.bool(True)
process.dumpGctDigis.emuGctInput = cms.untracked.InputTag("valGctDigis")
process.dumpGctDigis.doRegions = cms.untracked.bool(False)
process.dumpGctDigis.doInternEm = cms.untracked.bool(False)
process.dumpGctDigis.doEnergySums = cms.untracked.bool(True)
process.dumpGctDigis.doFibres = cms.untracked.bool(False)
process.dumpGctDigis.outFile = cms.untracked.string('gctDigis.txt')

# GCTErrorAnalyzer
process.load('L1Trigger.L1GctAnalyzer.gctErrorAnalyzer_cfi')

# Output ROOT file
process.TFileService = cms.Service("TFileService",
   fileName = cms.string( 'gctErrorAnalyzer.root' )
)

process.p = cms.Path(process.gctRaw*
                     process.l1GctHwDigis*
                     process.valGctDigis*
                     process.l1compare*
                     process.dumpGctDigis*
                     process.gctErrorAnalyzer)


process.output = cms.OutputModule( "PoolOutputModule",
                                   outputCommands = cms.untracked.vstring (
    "drop *",
    "keep *_*_*_GctPatternTester",
    ),
                                   fileName = cms.untracked.string( "gctPatternTester.root" ),
                                   )

process.out = cms.EndPath( process.output )
