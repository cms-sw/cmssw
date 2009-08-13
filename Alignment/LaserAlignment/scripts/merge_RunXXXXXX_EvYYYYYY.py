
import FWCore.ParameterSet.Config as cms

process = cms.Process( "laserAlignmentMerge" )

process.MessageLogger = cms.Service( "MessageLogger",
  cerr = cms.untracked.PSet(
    threshold = cms.untracked.string( 'ERROR' )
  ),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string( 'INFO' )
  ),
  destinations = cms.untracked.vstring( 'cout', 'cerr' )
)

process.source = cms.Source( "PoolSource",
  fileNames = cms.untracked.vstring(
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F1.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F2.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F3.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F4.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F5.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F6.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F7.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F8.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F9.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F10.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F11.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F12.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F13.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F14.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F15.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F16.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F17.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F18.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F19.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F20.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F21.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F22.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F23.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F24.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F25.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F26.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F27.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F28.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F29.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F30.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F31.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F32.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F33.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F34.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F35.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F36.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F37.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F38.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F39.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F40.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F41.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F42.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F43.root',
    'file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F44.root',
    ),
)

process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'GR09_31X_V5P::All'

process.eventFilter = cms.EDFilter("LaserAlignmentEventFilter",
    #pick Run and Event range
    RunFirst = cms.untracked.int32(XXXXXX),
    RunLast = cms.untracked.int32(XXXXXX), 
    EventFirst = cms.untracked.int32(YYYYYY),
    EventLast = cms.untracked.int32(000000)
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1)
#  input = cms.untracked.int32( 10000)
)

process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_laserAlignmentT0Producer_*_*'
  ),
  fileName = cms.untracked.string( '/tmp/aperiean/RunXXXXXX/TkAlLAS_RunXXXXXX_EvYYYYYY.root')
)

process.p = cms.Path( process.eventFilter+process.out)
