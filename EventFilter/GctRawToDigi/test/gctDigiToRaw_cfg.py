import FWCore.ParameterSet.Config as cms

process = cms.Process( "GctDigiToRaw" )

process.source = cms.Source( "PoolSource",
  fileNames = cms.untracked.vstring( "file:gctDigis.root" )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32 ( 10 ) )

process.load( "EventFilter/GctRawToDigi/gctDigiToRaw_cfi" )
process.gctDigiToRaw.verbose = cms.untracked.bool ( False )

process.dump = cms.OutputModule( "DumpFEDRawDataProduct",
  feds = cms.untracked.vint32( 745 ),
  dumpPayload = cms.untracked.bool( True )
)

process.p = cms.Path( process.gctDigiToRaw * process.dump )

process.output = cms.OutputModule( "PoolOutputModule",
  fileName = cms.untracked.string( "gctDigiToRaw.root" )
)

process.out = cms.EndPath( process.output )
