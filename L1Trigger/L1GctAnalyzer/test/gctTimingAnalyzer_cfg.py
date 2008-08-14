import FWCore.ParameterSet.Config as cms

process = cms.Process('TimingAnalyzer')

#Log messages
#import FWCore.MessageLogger.MessageLogger_cfi

#Logger
#process.MessageLogger = cms.Service (
#    destinations = cms.untracked.vstring ( "debug.log" ),
#    detailedInfo.txt = cms.untracked.PSet ( threshold = cms.untracked.string("INFO") ),
#    debugModules = cms.untracked.vstring ( "l1GctHwDigis" ),
#    #debugModules = cms.untracked.vstring ( "*" ),
#    #suppressWarning = cms.untracked.vstring ( "source", "l1GctHwDigis" )

#Input file
process.source = cms.Source ( "EmptySource" )

process.gctRaw = cms.OutputModule( "TextToRaw",
 filename = cms.untracked.string ( "multiplebx_crate14.dat" ),
)

process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 3564 ) )

# Unpacker
process.load('EventFilter/GctRawToDigi/l1GctHwDigis_cfi')
process.l1GctHwDigis.unpackFibres = cms.untracked.bool ( True )
process.l1GctHwDigis.verbose = cms.untracked.bool ( True )
process.l1GctHwDigis.inputLabel = cms.InputTag( "gctRaw" )
process.l1GctHwDigis.unpackInternEm = cms.untracked.bool ( True )

#Timing Analyzer
process.TimingAnalysis = cms.OutputModule( "GctTimingAnalyzer",
  isoEmSource = cms.untracked.InputTag("l1GctHwDigis","isoEm"),
  nonIsoEmSource = cms.untracked.InputTag("l1GctHwDigis","nonIsoEm"),
  internEmSource = cms.untracked.InputTag("l1GctHwDigis"),
  internJetSource = cms.untracked.InputTag("l1GctHwDigis"),
  cenJetsSource  = cms.untracked.InputTag("l1GctHwDigis","cenJets"),
  forJetsSource  = cms.untracked.InputTag("l1GctHwDigis","forJets"),
  tauJetsSource  = cms.untracked.InputTag("l1GctHwDigis","tauJets"),
  eSumsSource    = cms.untracked.InputTag("l1GctHwDigis"),
  fibreSource    = cms.untracked.InputTag("l1GctHwDigis"),
  rctSource      = cms.untracked.InputTag("l1GctHwDigis")
)

process.p = cms.Path ( process.gctRaw * process.l1GctHwDigis * process.TimingAnalysis )
