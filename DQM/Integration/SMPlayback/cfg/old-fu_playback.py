import FWCore.ParameterSet.Config as cms
process = cms.Process("HLT1")
 
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
   fileNames = cms.untracked.vstring('file:/home/dqmdevlocal/smPlayback/data/Data.00132440.0201.A.storageManager.10.0000.dat') 
    )   

process.playback = cms.EDFilter("PlaybackRawDataProvider")

process.p = cms.Path(process.playback)

