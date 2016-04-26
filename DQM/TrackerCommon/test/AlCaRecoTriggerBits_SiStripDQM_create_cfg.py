import FWCore.ParameterSet.Config as cms

process = cms.Process( "CREATE" )

process.load( "FWCore.MessageLogger.MessageLogger_cfi" )
process.MessageLogger.cerr = cms.untracked.PSet(
  placeholder = cms.untracked.bool( True )
)
process.MessageLogger.cout = cms.untracked.PSet(
  INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32( 1 )
  )
)

process.SiStripDQMCreate = cms.EDAnalyzer( "AlCaRecoTriggerBitsRcdUpdate"
, firstRunIOV = cms.uint32( 1 )
, lastRunIOV  = cms.int32( -1 )
, startEmpty = cms.bool( True )
, listNamesRemove = cms.vstring()
, triggerListsAdd = cms.VPSet(
    cms.PSet(
      listName = cms.string( 'Tracking_HLT' )
    , hltPaths = cms.vstring( 'HLT_ZeroBias_v*', 'HLT_BptxAnd_*' )
    ),
    cms.PSet(
      listName = cms.string( 'SiStrip_L1' )
    , hltPaths = cms.vstring( 'L1Tech_BPTX_plus_AND_minus.v0', 'L1_ZeroBias', 'L1_ExtCond_032' )
    ),
    cms.PSet(
      listName = cms.string( 'SiStrip_HLT' )
    , hltPaths = cms.vstring( 'HLT_ZeroBias_v*', 'HLT_HIZeroBias_v*', 'HLT_BptxAnd_*' )
    ),
  )
)

process.source = cms.Source( "EmptySource"
# , firstRun = cms.untracked.uint32( 1 )
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 1 )
)

import CondCore.DBCommon.CondDBSetup_cfi
# CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup.DBParameters.messageLevel = cms.untracked.int32( 3 )
process.PoolDBOutputService = cms.Service( "PoolDBOutputService"
, CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup
# , logconnect = cms.untracked.string( 'sqlite_file:AlCaRecoTriggerBits_SiStripDQM_create_log.db' )
, timetype = cms.untracked.string( 'runnumber' )
, connect  = cms.string( 'sqlite_file:AlCaRecoTriggerBits_TrackerDQM.db' )
, toPut    = cms.VPSet(
    cms.PSet(
      record = cms.string( 'AlCaRecoTriggerBitsRcd' )
    , tag    = cms.string( 'AlCaRecoTriggerBits_TrackerDQM_v1' )
    , label  = cms.untracked.string( 'TrackerDQMTrigger' ) 
    )
  )
)

process.p = cms.Path(
  process.SiStripDQMCreate
)


