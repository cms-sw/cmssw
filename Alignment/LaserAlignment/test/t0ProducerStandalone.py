
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
    fileNames = cms.untracked.vstring(
#      '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/659/4A6927B0-72AF-DD11-B589-000423D99AA2.root',
#      '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/659/5A4CE8B0-72AF-DD11-8C63-000423D6AF24.root',
      '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/659/B00E67AC-8AAF-DD11-9F9B-000423D98BC4.root',
      '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/659/CAA4D81E-74AF-DD11-94D1-000423D99996.root',
      '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/659/CE3958DD-74AF-DD11-A328-000423D99614.root'
      #'/store/data/Commissioning08/TestEnables/RAW/v1/000/070/659/FAEEFA64-73AF-DD11-B0EA-000423D99896.root' #DONE
    )
)

process.load( "EventFilter.SiStripRawToDigi.SiStripDigis_cfi" )
process.siStripDigis.ProductLabel = "hltCalibrationRaw"

process.load( "CondCore.DBCommon.CondDBSetup_cfi" )
process.allSource = cms.ESSource( "PoolDBESSource",
  process.CondDBSetup,
  connect = cms.string( 'frontier://FrontierProd/CMS_COND_20X_GLOBALTAG' ),
  globaltag = cms.string( 'IDEAL_v2::All' )
                                    
)


# multiple sets can be given, only those will be output
process.load( "Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi" )
process.laserAlignmentT0Producer.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiType = cms.string( 'Processed' ),
    DigiProducer = cms.string( 'siStripDigis' )
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
  fileName = cms.untracked.string( '/afs/cern.ch/user/o/olzem/scratch0/LaserEvents.70659_345.root' )
)


# process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.seqDigitization = cms.Path( process.siStripDigis + process.laserAlignmentT0Producer )

process.outputPath = cms.EndPath( process.out )



