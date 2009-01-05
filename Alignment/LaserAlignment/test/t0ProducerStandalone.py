
import FWCore.ParameterSet.Config as cms

process = cms.Process("laserAlignmentT0ProducerProcess")

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
    fileNames = cms.untracked.vstring( 'file:/afs/cern.ch/user/o/olzem/scratch0/LaserEvents.SIM-DIGI.root' )
)

process.load( "EventFilter.SiStripRawToDigi.SiStripDigis_cfi" )

 
process.load( "Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi" )
process.laserAlignmentT0Producer.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'VirginRaw' ),
    DigiType = cms.string( 'Raw' ),
    DigiProducer = cms.string( 'simSiStripDigis' )
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)

process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_laserAlignmentT0Producer_*_*'
  ),
  fileName = cms.untracked.string( '/afs/cern.ch/user/o/olzem/scratch0/LaserEvents.ALCARECO_210.root' )
)


process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.seqDigitization = cms.Sequence( process.SiStripDigis )
process.pReconstruction = cms.Path( process.laserAlignmentT0Producer )

process.outputPath = cms.EndPath( process.out )



