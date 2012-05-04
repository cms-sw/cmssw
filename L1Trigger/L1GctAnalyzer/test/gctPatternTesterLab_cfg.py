#
# This configuration is intended to be used to check pattern tests with the GCT emulator 
#
# Alex Tapper 8/9/10
# Edited by Georgia to configure for Pattern tests with 904 sys

process = cms.Process('GctPatternTester')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')
process.MessageLogger = cms.Service ("MessageLogger", 
	destinations = cms.untracked.vstring( "detailedInfo.txt" ),
	threshold = cms.untracked.string ( 'WARNING' )
)

process.source = cms.Source ( "EmptySource" )

#One orbit of data is default for capture 
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 10000 ) )

#Input captured ascii file
process.gctRaw = cms.EDProducer( "TextToRaw",
                                   filename = cms.untracked.string (
    "patternCapture_ts__2012_04_26__12h02m20s.txt"), # Jet seed off                                
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
process.valGctDigis.useImprovedTauAlgorithm = cms.bool(False)
## Added for Lab sys:
process.valGctDigis.hardwareTest = cms.bool(True)
process.valGctDigis.jetLeafMask = cms.untracked.uint32(31) # 0x1f

# HF masking:
process.L1GctConfigProducers.MEtEtaMask = cms.uint32(0x3C000F)
process.L1GctConfigProducers.TEtEtaMask = cms.uint32(0x3C000F)
process.L1GctConfigProducers.MHtEtaMask = cms.uint32(0x3E001F)
process.L1GctConfigProducers.HtEtaMask = cms.uint32(0x3E001F)

# GCT unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.inputLabel = cms.InputTag( "gctRaw" )
process.l1GctHwDigis.verbose = cms.untracked.bool(False)
process.l1GctHwDigis.unpackerVersion = cms.uint32(3)

# L1Comparator
process.load('L1Trigger.HardwareValidation.L1Comparator_cfi')
process.l1compare.GCTsourceData = cms.InputTag("l1GctHwDigis")
process.l1compare.COMPARE_COLLS = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
process.l1compare.DumpMode = cms.untracked.int32(1)
process.l1compare.DumpFile = cms.untracked.string('dump.txt')
process.l1compare.VerboseFlag = cms.untracked.int32(0)
# L1Comparator Filter    
process.load('L1Trigger.HardwareValidation.L1DEFilter_cfi')
process.l1defilter.DataEmulCompareSource = cms.InputTag("l1compare")
process.l1defilter.FlagSystems = cms.untracked.vuint32(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)

# Dump GCT digis
process.load('L1Trigger.L1GctAnalyzer.dumpGctDigis_cfi')
process.dumpGctDigis.doRctEm = cms.untracked.bool(True)
process.dumpGctDigis.doEm = cms.untracked.bool(True)
process.dumpGctDigis.emuRctInput = cms.untracked.InputTag("l1GctHwDigis")
process.dumpGctDigis.doJets = cms.untracked.bool(True)
process.dumpGctDigis.doEmulated = cms.untracked.bool(True)
process.dumpGctDigis.emuGctInput = cms.untracked.InputTag("valGctDigis")
process.dumpGctDigis.doRegions = cms.untracked.bool(True)
process.dumpGctDigis.doInternEm = cms.untracked.bool(True)
process.dumpGctDigis.doEnergySums = cms.untracked.bool(True)
process.dumpGctDigis.doFibres = cms.untracked.bool(False)
process.dumpGctDigis.outFile = cms.untracked.string('gctDigis.txt')

# GCTErrorAnalyzer
process.load('L1Trigger.L1GctAnalyzer.gctErrorAnalyzer_cfi')
process.gctErrorAnalyzer.useSys = cms.untracked.string("Lab") ## Lab (904) system
process.gctErrorAnalyzer.doExtraMissingHTDebug = cms.untracked.bool(True)

# Output ROOT file
process.TFileService = cms.Service("TFileService",
   fileName = cms.string( 'gctErrorAnalyzer.root' )
)

process.p = cms.Path(process.gctRaw*
                     process.l1GctHwDigis*
                     process.valGctDigis*
                     process.l1compare*
#                     ~process.l1defilter*
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
