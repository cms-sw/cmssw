import FWCore.ParameterSet.Config as cms

process = cms.Process( "GctRawToDigi" )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service ( "MessageLogger",
#  destinations = cms.untracked.vstring ( "debug.log" ),
#  debug = cms.untracked.PSet ( threshold = cms.untracked.string ( "DEBUG" ) ),
#  debugModules = cms.untracked.vstring ( "DumpFedRawDataProduct", "TextToRaw", "GctRawToDigi" )
#)

process.source = cms.Source( "NewEventStreamFileReader",
  fileNames = cms.untracked.vstring(
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0001.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0001.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0001.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0001.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0002.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0002.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0002.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0002.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0003.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0003.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0003.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0003.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0004.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0004.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0004.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0004.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0005.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0005.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0005.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0005.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0006.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0006.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0006.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0006.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0007.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0007.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0007.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0007.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0008.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0008.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0008.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0008.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0009.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0009.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0009.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0009.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0010.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0010.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0010.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0010.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0011.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0011.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0011.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0011.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0012.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0012.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0012.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0012.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0013.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0013.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0013.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0013.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0014.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0014.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0014.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0014.Random.storageManager.3.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0015.Random.storageManager.0.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0015.Random.storageManager.1.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0015.Random.storageManager.2.0000.dat',
'/store/data/GlobalCruzet3MW33/Random/000/056/632/GlobalCruzet3MW33.00056632.0015.Random.storageManager.3.0000.dat'

  )
)

#process.source = cms.Source( "PoolSource",
#  fileNames = cms.untracked.vstring( "file:gctRaw.root" )
#)

process.dumpRaw = cms.OutputModule( "DumpFEDRawDataProduct",
  feds = cms.untracked.vint32( 745 ),
  dumpPayload = cms.untracked.bool( True )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1000 ) )

process.load( "EventFilter/GctRawToDigi/l1GctHwDigis_cfi" )
process.l1GctHwDigis.inputLabel = cms.InputTag( "source" )

process.p = cms.Path( process.dumpRaw * process.l1GctHwDigis )

process.output = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring ( 
    "keep *",
#    "keep *_l1GctHwDigis_*_*",
#    "keep *_gctDigiToRaw_*_*"
  ),
  
  fileName = cms.untracked.string( "gctDigis.root" )

)
process.out = cms.EndPath( process.output )
