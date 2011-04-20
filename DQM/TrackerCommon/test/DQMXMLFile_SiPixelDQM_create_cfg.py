import FWCore.ParameterSet.Config as cms

process = cms.Process( "CREATE" )

process.MessageLogger=cms.Service( "MessageLogger"
, cout = cms.untracked.PSet(
    threshold = cms.untracked.string( 'INFO' )
  )
, destinations = cms.untracked.vstring(
    'cout'
  )
)

process.source = cms.Source( "EmptyIOVSource"
, timetype   = cms.string( 'runnumber' )
, firstValue = cms.uint64( 1 )
, lastValue  = cms.uint64( 1 )
, interval   = cms.uint64( 1 )
)

process.dqmXmlFileTest = cms.EDAnalyzer( "DQMXMLFilePopConAnalyzer"
, record          = cms.string( 'FileBlob' )
, loggingOn       = cms.untracked.bool( True )
, SinceAppendMode = cms.bool( False )
, Source          = cms.PSet(
    XMLFile    = cms.untracked.string( '/afs/cern.ch/cms/sw/slc5_ia32_gcc434/cms/cmssw/CMSSW_3_7_0/src/DQM/SiPixelMonitorClient/test/sipixel_tier0_qualitytest.xml' )
  , firstSince = cms.untracked.uint64( 1 )
  , debug      = cms.untracked.bool( False )
  , zip        = cms.untracked.bool( False )
  )
)

process.load( "CondCore.DBCommon.CondDBCommon_cfi" )
process.CondDBCommon.connect          = cms.string( 'sqlite_file:DQMXMLFile_SiPixelDQM.db' )
process.CondDBCommon.BlobStreamerName = cms.untracked.string( 'TBufferBlobStreamingService' )
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string( '' )
# process.CondDBCommon.DBParameters.messageLevel       = cms.untracked.int32( 3 )

process.PoolDBOutputService = cms.Service( "PoolDBOutputService"
, process.CondDBCommon
, logconnect = cms.untracked.string( 'sqlite_file:DQMXMLFile_SiPixelDQM_create_log.db' )
, timetype   = cms.untracked.string( 'runnumber' )
, toPut      = cms.VPSet(
    cms.PSet(
      record = cms.string( 'FileBlob' )
    , tag    = cms.string( 'DQMXMLFile_SiPixelDQM_v0_test' )
    )
  )
)

process.p = cms.Path(
  process.dqmXmlFileTest
)
