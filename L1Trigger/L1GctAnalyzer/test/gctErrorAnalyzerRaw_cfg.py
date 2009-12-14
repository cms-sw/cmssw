import FWCore.ParameterSet.Config as cms

process = cms.Process("GCTAnalyzerTest")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service ("MessageLogger", 
	destinations = cms.untracked.vstring( "detailedInfo.txt" ),
	threshold = cms.untracked.string ( 'WARNING' )
)

process.source = cms.Source ( "EmptySource" )

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 3563 ) ) #use 3563 for whole orbit

# raw data
process.gctRaw = cms.OutputModule( "TextToRaw",
  filename = cms.untracked.string("slinkOutput.txt"),
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
process.load('L1Trigger.Configuration.L1StartupConfig_cff')
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
process.valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
process.valGctDigis.writeInternalData = cms.bool(True)
process.valGctDigis.inputLabel = cms.InputTag( "l1GctHwDigis" )
process.valGctDigis.preSamples = cms.uint32(0)
process.valGctDigis.postSamples = cms.uint32(0)

# my analyzer
process.TFileService = cms.Service("TFileService",
   fileName = cms.string( 'raw_gctAnalyzer.root' )
)

# GCT Analyzer
process.analyzer = cms.EDAnalyzer('GctErrorAnalyzer',
   #Multiple BX Flags
   doRCTMBx = cms.untracked.bool(False),
   doEmuMBx = cms.untracked.bool(True),
   doGCTMBx = cms.untracked.bool(True),
   #Plot + Debug Info Flags
   doRCT = cms.untracked.bool(True),
   doEg = cms.untracked.bool(True),
   doIsoDebug = cms.untracked.bool(True),
   doNonIsoDebug = cms.untracked.bool(True),
   doJets = cms.untracked.bool(True),
   doCenJetsDebug = cms.untracked.bool(True),
   doTauJetsDebug = cms.untracked.bool(True),
   doForJetsDebug = cms.untracked.bool(True),
   doHF = cms.untracked.bool(True),
   doRingSumDebug = cms.untracked.bool(True),
   doBitCountDebug = cms.untracked.bool(True),
   doTotalEnergySums = cms.untracked.bool(True),
   doTotalHtDebug = cms.untracked.bool(True),
   doTotalEtDebug = cms.untracked.bool(True),
   doMissingEnergySums = cms.untracked.bool(True),
   doMissingETDebug = cms.untracked.bool(True),
   doMissingHTDebug = cms.untracked.bool(True),
   #Labels to use for data and emulator
   dataTag = cms.untracked.InputTag("l1GctHwDigis"),
   emuTag = cms.untracked.InputTag("valGctDigis"),
   #Nominally, the following parameters should NOT be changed
   RCTTrigBx = cms.untracked.int32(0),
   EmuTrigBx = cms.untracked.int32(0),
   GCTTrigBx = cms.untracked.int32(0),
)

process.defaultPath = cms.Sequence ( 
# Text to Raw
process.gctRaw *
# Unpacker
process.l1GctHwDigis *
# Emulator
process.valGctDigis *
# GCTAnalyzer
process.analyzer
)

process.p = cms.Path(process.defaultPath)
