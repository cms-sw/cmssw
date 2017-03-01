import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")

## Options
process.options = cms.untracked.PSet(
  wantSummary      = cms.untracked.bool( True )
, allowUnscheduled = cms.untracked.bool( True )
)

## Messaging
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.Tracer = cms.Service("Tracer")

## Conditions
process.load( "Configuration.Geometry.GeometryRecoDB_cff" )
process.load( "Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff" )
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag( process.GlobalTag, 'auto:com10_8E33v2' )

## Input
from PhysicsTools.PatAlgos.patInputFiles_cff import filesSingleMuRECO
process.source = cms.Source(
  "PoolSource"
, fileNames = filesSingleMuRECO
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 100 )
)

## Output
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out = cms.OutputModule(
  "PoolOutputModule"
, fileName = cms.untracked.string( 'patTuple_data.root' )
, outputCommands = cms.untracked.vstring(
    *patEventContentNoCleaning
  )
)
process.out.outputCommands += [ 'drop recoGenJets_*_*_*' ]
process.outpath = cms.EndPath(
  process.out
)

## Processing
process.load( "PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff" )
process.load( "PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff" )
# for data:
from PhysicsTools.PatAlgos.tools.coreTools import runOnData
runOnData( process )
