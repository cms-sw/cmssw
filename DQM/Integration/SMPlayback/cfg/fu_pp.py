import FWCore.ParameterSet.Config as cms
process = cms.Process("HLT1")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound','TooManyProducts','TooFewProducts'),
    makeTriggerResults = cms.untracked.bool(True)
    )

 
process.MessageLogger = cms.Service("MessageLogger",
  destinations = cms.untracked.vstring('cout'),
#   log4cplus = cms.untracked.PSet(threshold = cms.untracked.string('WARNING')),
  cout = cms.untracked.PSet(    threshold = cms.untracked.string('WARNING')
    )
 )
 
process.source = cms.Source("AdaptorConfig")
 
process.source = cms.Source("SiteLocalConfigService")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("NewEventStreamFileReader", 
   loop = cms.untracked.bool(True),
   eventsPerLS = cms.untracked.int32(1000),
   fileNames = cms.untracked.vstring(["file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0471.A.storageManager.00.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0471.A.storageManager.01.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0471.A.storageManager.02.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0471.A.storageManager.03.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0471.A.storageManager.04.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0471.A.storageManager.05.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0471.A.storageManager.06.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0471.A.storageManager.07.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0472.A.storageManager.00.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0472.A.storageManager.01.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0472.A.storageManager.02.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0472.A.storageManager.03.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0472.A.storageManager.04.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0472.A.storageManager.05.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0472.A.storageManager.06.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0472.A.storageManager.07.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0473.A.storageManager.00.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0473.A.storageManager.01.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0473.A.storageManager.02.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0473.A.storageManager.03.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0473.A.storageManager.04.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0473.A.storageManager.05.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0473.A.storageManager.06.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0473.A.storageManager.07.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0474.A.storageManager.00.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0474.A.storageManager.01.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0474.A.storageManager.02.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0474.A.storageManager.03.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0474.A.storageManager.04.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0474.A.storageManager.05.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0474.A.storageManager.06.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0474.A.storageManager.07.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0475.A.storageManager.00.0000.dat"
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0475.A.storageManager.01.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0475.A.storageManager.02.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0475.A.storageManager.03.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0475.A.storageManager.04.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0475.A.storageManager.05.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0475.A.storageManager.06.0000.dat",
"file:/home/dqmdevlocal/smPlayback2/data/Data.00135528.0475.A.storageManager.07.0000.dat"] 
)   
)
################################################################cd ../
process.pre1 = cms.EDFilter("Prescaler",
                          prescaleFactor = cms.int32(1),
                          prescaleOffset = cms.int32(0)
                          )


#process.load("HLTrigger.special.HLTHcalCalibTypeFilter_cfi")
#process.hltHcalCalibTypeFilter.CalibTypes=cms.vint32( 1,2,3,4,5,6 )
#process.HLT_HcalCalibration = cms.Path(process.hltHcalCalibTypeFilter)

process.playbackPath4DQM = cms.Path( process.pre1)

process.hltOutputDQM = cms.OutputModule("ShmStreamConsumer",
               fakeHLTPaths = cms.untracked.bool(True),
               max_event_size = cms.untracked.int32(7000000),
               max_queue_depth = cms.untracked.int32(5),
               use_compression = cms.untracked.bool(True),
               compression_level = cms.untracked.int32(1),
               SelectEvents = cms.untracked.PSet(
                 SelectEvents = cms.vstring('*')
                 )
         )

process.hltOutputHLTDQM = cms.OutputModule("ShmStreamConsumer",
               fakeHLTPaths = cms.untracked.bool(True),
               max_event_size = cms.untracked.int32(7000000),
               max_queue_depth = cms.untracked.int32(5),
               use_compression = cms.untracked.bool(True),
               compression_level = cms.untracked.int32(1),
               SelectEvents = cms.untracked.PSet(
                 SelectEvents = cms.vstring('*')
                 )
            )

process.hltOutputEventDisplay = cms.OutputModule("ShmStreamConsumer",
               fakeHLTPaths = cms.untracked.bool(True),
               max_event_size = cms.untracked.int32(7000000),
               max_queue_depth = cms.untracked.int32(5),
               use_compression = cms.untracked.bool(True),
               compression_level = cms.untracked.int32(1),
               SelectEvents = cms.untracked.PSet(
                 SelectEvents = cms.vstring('*')
                 )
            )

process.e = cms.EndPath(process.hltOutputDQM*process.hltOutputHLTDQM*process.hltOutputEventDisplay)


