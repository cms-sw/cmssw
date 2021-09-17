import FWCore.ParameterSet.Config as cms

process = cms.Process( "CREATE" )

process.load( "FWCore.MessageLogger.MessageLogger_cfi" )
process.MessageLogger.cerr.enable = False
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
      listName = cms.string( 'SiStripDQM_L1' )
    , hltPaths = cms.vstring(
        'NOT L1Tech_BSC_halo_beam2_inner.v0'                                   # NOT 36
      , 'NOT L1Tech_BSC_halo_beam2_outer.v0'                                   # NOT 37
      , 'NOT L1Tech_BSC_halo_beam1_inner.v0'                                   # NOT 38
      , 'NOT L1Tech_BSC_halo_beam1_outer.v0'                                   # NOT 39
      , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'  # NOT (42 AND NOT 43)
      , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)'  # NOT (43 AND NOT 42)
      )
    )
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
, connect  = cms.string( 'sqlite_file:AlCaRecoTriggerBits_SiStripDQM.db' )
, toPut    = cms.VPSet(
    cms.PSet(
      record = cms.string( 'AlCaRecoTriggerBitsRcd' )
    , tag    = cms.string( 'AlCaRecoTriggerBits_SiStripDQM_v2_test' )
    )
  )
)

process.p = cms.Path(
  process.SiStripDQMCreate
)


