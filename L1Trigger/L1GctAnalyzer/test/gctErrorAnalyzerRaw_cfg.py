import FWCore.ParameterSet.Config as cms

process = cms.Process("GCTAnalyzerTest")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(100000)
#process.MessageLogger = cms.Service ("MessageLogger", 
#	destinations = cms.untracked.vstring( "detailedInfo.txt" ),
#	threshold = cms.untracked.string ( 'WARNING' )
#)

process.source = cms.Source ( "EmptySource" )

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 3563 ) ) #use 3563 for whole orbit

# raw data
process.gctRaw = cms.EDProducer( "TextToRaw",
  filename = cms.untracked.string("patternCaptureOrbit_hwtest__2011_02_18__10h01m55s_HfInc.txt"),
   GctFedId = cms.untracked.int32 ( 745 ),
  FileEventOffset = cms.untracked.int32 ( 0 )
)

# unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.unpackerVersion = cms.uint32(3)
#process.l1GctHwDigis.unpackSharedRegions = cms.bool ( True )
process.l1GctHwDigis.numberOfGctSamplesToUnpack = cms.uint32(5)
process.l1GctHwDigis.hltMode = cms.bool( False )
process.l1GctHwDigis.inputLabel = cms.InputTag( "gctRaw" )
process.l1GctHwDigis.verbose = cms.untracked.bool ( False )
process.l1GctHwDigis.unpackFibres = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternEm = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternJets = cms.untracked.bool ( True )

# GCT emulator
process.load('L1Trigger.L1GctAnalyzer.gctPatternTestConfig_cff')
#process.load('L1TriggerConfig.GctConfigProducers.l1GctConfig_cfi')
#process.load('L1Trigger.Configuration.L1StartupConfig_cff')
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
process.valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
process.valGctDigis.writeInternalData = cms.bool(True)
process.valGctDigis.inputLabel = cms.InputTag( "l1GctHwDigis" )
process.valGctDigis.preSamples = cms.uint32(0)
process.valGctDigis.postSamples = cms.uint32(0)

process.valGctDigis.hardwareTest = cms.bool(True) ## specific to "Lab" configuration
process.valGctDigis.jetLeafMask = cms.untracked.uint32(0x1f) ## specific to "Lab" configuration


## HF exclusion
#process.L1GctConfigProducers.MEtEtaMask = cms.uint32(0x3C000F)
#process.L1GctConfigProducers.TEtEtaMask = cms.uint32(0x3C000F)
#process.L1GctConfigProducers.MHtEtaMask = cms.uint32(0x3E001F)
#process.L1GctConfigProducers.HtEtaMask = cms.uint32(0x3E001F)

# my analyzer
process.TFileService = cms.Service("TFileService",
   fileName = cms.string( 'raw_gctAnalyzer.root' )
)

# GCT Error Analyzer
process.load('L1Trigger.L1GctAnalyzer.gctErrorAnalyzer_cfi')
process.gctErrorAnalyzer.doEmuMBx = cms.untracked.bool(True)
process.gctErrorAnalyzer.doGCTMBx = cms.untracked.bool(True)
process.gctErrorAnalyzer.doExtraMissingHTDebug = cms.untracked.bool(False)
process.gctErrorAnalyzer.useSys = cms.untracked.string("Lab") ## specific to "Lab" configuration

process.load("L1TriggerConfig.GctConfigProducers.l1GctConfigDump_cfi")
 

process.defaultPath = cms.Sequence ( 
# Text to Raw
process.gctRaw *
# Unpacker
process.l1GctHwDigis *
# Emulator
process.valGctDigis *
# GCTErrorAnalyzer
process.gctErrorAnalyzer #*

#process.l1GctConfigDump
)

process.p = cms.Path(process.defaultPath)
