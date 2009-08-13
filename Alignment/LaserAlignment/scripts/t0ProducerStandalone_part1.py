
import FWCore.ParameterSet.Config as cms

process = cms.Process( "laserAlignmentT0ProducerProcess" )

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
