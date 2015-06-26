import FWCore.ParameterSet.Config as cms

process = cms.Process( "RctRawToDigi" )

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(100)



process.dumpRaw = cms.OutputModule( "DumpFEDRawDataProduct",
  feds = cms.untracked.vint32( 745 ),
  dumpPayload = cms.untracked.bool( True )
)

# unpacker
process.load( "EventFilter.RctRawToDigi.l1RctHwDigis_cfi" )
process.l1RctHwDigis.inputLabel = cms.InputTag( "rawDataCollector" )
process.l1RctHwDigis.verbose = cms.untracked.bool( True )
process.l1RctHwDigis.rctFedId = cms.untracked.int32( 745 )

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.output = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring ( 
    "keep *",
#    "keep *_l1RctHwDigis_*_*",
#    "keep *_rctDigiToRaw_*_*"
  ),
  
  fileName = cms.untracked.string( "rctDigis.root" )

)

process.p = cms.Path( 
    process.l1RctHwDigis + process.dumpRaw
 )

process.out = cms.EndPath( process.output )

# input
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 100 ) )

process.source = cms.Source ( "PoolSource",
   fileNames = cms.untracked.vstring(
    'file:6AB2D8F5-4E1B-E511-9093-02163E013944.root'
  )
)

#process.source = cms.Source( "PoolSource",
#  fileNames = cms.untracked.vstring( "file:rctRaw.root" )
#)
