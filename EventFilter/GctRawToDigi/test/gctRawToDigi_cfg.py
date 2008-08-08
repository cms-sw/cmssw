import FWCore.ParameterSet.Config as cms

process = cms.Process( "GctRawToDigi" )

import FWCore.MessageLogger.MessageLogger_cfi
process.MessageLogger = cms.Service ( 
  destinations = cms.untracked.vstring ( "debug.log" ),
  debuglog = cms.untracked.PSet ( threshold = cms.untracked.string ( "DEBUG" ) ),
  debugModules = cms.untracked.vstring ( "DumpFedRawDataProduct", "TextToRaw", "GctRawToDigi" )
)

process.source = cms.Source( "NewEventStreamFileReader",
  fileNames = cms.untracked.vstring( "file:gctDigiToRaw.root" )
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
