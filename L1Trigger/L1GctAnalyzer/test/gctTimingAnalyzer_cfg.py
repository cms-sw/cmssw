import FWCore.ParameterSet.Config as cms

process = cms.Process('TimingAnalyzer')

#Log messages
#import FWCore.MessageLogger.MessageLogger_cfi

#Logger
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    debuglog = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(

        )
    )
)

#Input file
process.source = cms.Source ( "EmptySource" )

process.gctRaw = cms.OutputModule( "TextToRaw",
 filename = cms.untracked.string ( "bx94.txt" ),
)

process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 1 ) )

# Unpacker
process.load('EventFilter/GctRawToDigi/l1GctHwDigis_cfi')
process.l1GctHwDigis.unpackFibres = cms.untracked.bool ( True )
process.l1GctHwDigis.verbose = cms.untracked.bool ( True )
process.l1GctHwDigis.inputLabel = cms.InputTag( "gctRaw" )
process.l1GctHwDigis.unpackInternEm = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternJets = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternESums = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternHF = cms.untracked.bool ( True )

#Timing Analyzer
process.TimingAnalysis = cms.OutputModule( "GctTimingAnalyzer",
  isoEmSource    = cms.untracked.InputTag("l1GctHwDigis","isoEm"),
  nonIsoEmSource = cms.untracked.InputTag("l1GctHwDigis","nonIsoEm"),
  gctSource      = cms.untracked.InputTag("l1GctHwDigis"),
  cenJetsSource  = cms.untracked.InputTag("l1GctHwDigis","cenJets"),
  forJetsSource  = cms.untracked.InputTag("l1GctHwDigis","forJets"),
  tauJetsSource  = cms.untracked.InputTag("l1GctHwDigis","tauJets"),
  doInternal     = cms.untracked.bool(True),
  doElectrons    = cms.untracked.bool(True),
  doJets         = cms.untracked.bool(True),
  doHFRings      = cms.untracked.bool(True),
  doESums        = cms.untracked.bool(True)
)

process.p = cms.Path ( process.gctRaw * process.l1GctHwDigis *process.TimingAnalysis )

process.output = cms.OutputModule ( "PoolOutputModule",
  outputCommands = cms.untracked.vstring ( 
    "drop *",
    "keep *_l1GctHwDigis_*_*",
    "keep *_gctRaw_*_*"
  ),

  fileName = cms.untracked.string ( "gctDigis.root" )

)
process.out = cms.EndPath(process.output)
