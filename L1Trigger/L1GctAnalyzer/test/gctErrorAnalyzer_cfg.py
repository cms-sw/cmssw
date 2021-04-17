import FWCore.ParameterSet.Config as cms

process = cms.Process("GCTAnalyzerTest")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        detailedInfo = cms.untracked.PSet(
            extension = cms.untracked.string('txt')
        )
    ),
    threshold = cms.untracked.string('WARNING')
)

process.source = cms.Source ( "PoolSource",
#   fileNames = cms.untracked.vstring('/store/data/Commissioning09/Calo/RAW/v1/000/096/889/FCF95AAE-0E44-DE11-BE87-000423D8FA38.root')
#   fileNames = cms.untracked.vstring('/store/data/Commissioning09/Calo/RAW/v2/000/100/329/FE4BEE2D-815B-DE11-8C71-001D09F2AD84.root')
    fileNames = cms.untracked.vstring('/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/602/08C1CF7A-D940-DF11-91F1-00E08178C01B.root')
)

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 100 ) )

# unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.unpackerVersion = cms.uint32(3)
#process.l1GctHwDigis.unpackSharedRegions = cms.bool ( True )
process.l1GctHwDigis.inputLabel = cms.InputTag( "source" )
process.l1GctHwDigis.numberOfGctSamplesToUnpack = cms.uint32(5)
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
process.valGctDigis.useImprovedTauAlgorithm = cms.bool(False)
process.valGctDigis.preSamples = cms.uint32(0)
process.valGctDigis.postSamples = cms.uint32(0)

# my analyzer
process.TFileService = cms.Service("TFileService",
   fileName = cms.string( 'gctAnalyzer.root' )
)

process.load('L1Trigger.L1GctAnalyzer.gctErrorAnalyzer_cfi')

process.defaultPath = cms.Sequence ( 
# Unpacker
process.l1GctHwDigis *
# Emulator
process.valGctDigis *
# GCTErrorAnalyzer
process.gctErrorAnalyzer
)

process.p = cms.Path(process.defaultPath)
