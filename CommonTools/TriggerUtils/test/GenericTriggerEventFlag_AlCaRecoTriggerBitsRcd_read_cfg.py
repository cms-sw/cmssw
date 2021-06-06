import FWCore.ParameterSet.Config as cms

process = cms.Process( "READ" )

process.load( "FWCore.MessageLogger.MessageLogger_cfi" )
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
  INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32( 250 )
  )
)

process.source = cms.Source( "EmptySource"
, numberEventsInRun = cms.untracked.uint32( 1 ) # do not change!
, firstRun          = cms.untracked.uint32( 123000 )
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 2000 )
)

import CondCore.DBCommon.CondDBSetup_cfi
process.dbInput = cms.ESSource( "PoolDBESSource"
, CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup
, connect = cms.string( 'sqlite_file:GenericTriggerEventFlag_AlCaRecoTriggerBits.db' )
, toGet   = cms.VPSet(
    cms.PSet(
      record = cms.string( 'AlCaRecoTriggerBitsRcd' )
    , tag    = cms.string( 'AlCaRecoTriggerBits_v0_test' )
    )
  )
)

process.AlCaRecoTriggerBitsRcdRead = cms.EDAnalyzer( "AlCaRecoTriggerBitsRcdRead"
, outputType  = cms.untracked.string( 'text' )
, rawFileName = cms.untracked.string( 'GenericTriggerEventFlag_AlCaRecoTriggerBits' )
)

process.p = cms.Path(
  process.AlCaRecoTriggerBitsRcdRead
)
