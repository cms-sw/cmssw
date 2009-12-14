import FWCore.ParameterSet.Config as cms

process = cms.Process("GCTAnalyzerTest")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service ("MessageLogger", 
	destinations = cms.untracked.vstring( "detailedInfo.txt" ),
	threshold = cms.untracked.string ( 'WARNING' )
)

process.source = cms.Source ( "PoolSource",
#   fileNames = cms.untracked.vstring('/store/data/Commissioning09/Calo/RAW/v1/000/096/889/FCF95AAE-0E44-DE11-BE87-000423D8FA38.root')
    fileNames = cms.untracked.vstring('/store/data/Commissioning09/Calo/RAW/v2/000/100/329/FE4BEE2D-815B-DE11-8C71-001D09F2AD84.root')
)

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 100 ) )

# unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.unpackerVersion = cms.uint32(3)
#process.l1GctHwDigis.unpackSharedRegions = cms.bool ( True )
process.l1GctHwDigis.inputLabel = cms.InputTag( "source" )
process.l1GctHwDigis.numberOfGctSamplesToUnpack = cms.uint32(1)
process.l1GctHwDigis.hltMode = cms.bool( False )
process.l1GctHwDigis.verbose = cms.untracked.bool ( False )
process.l1GctHwDigis.unpackFibres = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternEm = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternJets = cms.untracked.bool ( True )

# GCT emulator
process.load('L1Trigger.Configuration.L1StartupConfig_cff')
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
process.valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
process.valGctDigis.inputLabel = cms.InputTag( "l1GctHwDigis" )
process.valGctDigis.writeInternalData = cms.bool(True)
process.valGctDigis.preSamples = cms.uint32(0)
process.valGctDigis.postSamples = cms.uint32(0)

# my analyzer
process.TFileService = cms.Service("TFileService",
   fileName = cms.string( 'gctAnalyzer.root' )
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
     doTotalHtDebug = cms.untracked.bool(False),
     doTotalEtDebug = cms.untracked.bool(False),
   doMissingEnergySums = cms.untracked.bool(True),
     doMissingETDebug = cms.untracked.bool(False),
     doMissingHTDebug = cms.untracked.bool(False),
   #Labels to use for data and emulator
   dataTag = cms.untracked.InputTag("l1GctHwDigis"),
   emuTag = cms.untracked.InputTag("valGctDigis"),
   #Nominally, the following parameters should NOT be changed
   RCTTrigBx = cms.untracked.int32(0),
   EmuTrigBx = cms.untracked.int32(0),
   GCTTrigBx = cms.untracked.int32(0),
)

process.defaultPath = cms.Sequence ( 
# Unpacker
process.l1GctHwDigis *
# Emulator
process.valGctDigis *
# GCTAnalyzer
process.analyzer
)

process.p = cms.Path(process.defaultPath)
