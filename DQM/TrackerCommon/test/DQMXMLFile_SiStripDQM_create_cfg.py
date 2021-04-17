from __future__ import print_function
import FWCore.ParameterSet.Config as cms

import os

process = cms.Process( "CREATE" )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
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
    XMLFile    = cms.untracked.string( os.getenv( 'CMSSW_RELEASE_BASE' ) + '/src/DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_tier0.xml' )
  , firstSince = cms.untracked.uint64( 1 )
  , debug      = cms.untracked.bool( False )
  , zip        = cms.untracked.bool( False )
  )
)
print("Used XML file: " + process.dqmXmlFileTest.Source.XMLFile.pythonValue())

process.load( "CondCore.DBCommon.CondDBCommon_cfi" )
process.CondDBCommon.connect          = cms.string( 'sqlite_file:DQMXMLFile_SiStripDQM.db' )
process.CondDBCommon.BlobStreamerName = cms.untracked.string( 'TBufferBlobStreamingService' )
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string( '' )
# process.CondDBCommon.DBParameters.messageLevel       = cms.untracked.int32( 3 )

process.PoolDBOutputService = cms.Service( "PoolDBOutputService"
, process.CondDBCommon
, logconnect = cms.untracked.string( 'sqlite_file:DQMXMLFile_SiStripDQM_create_log.db' )
, timetype   = cms.untracked.string( 'runnumber' )
, toPut      = cms.VPSet(
    cms.PSet(
      record = cms.string( 'FileBlob' )
    , tag    = cms.string( 'DQMXMLFile_SiStripDQM_v1_test' )
    )
  )
)

process.p = cms.Path(
  process.dqmXmlFileTest
)
