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
process.load( "Configuration.Geometry.GeometryIdeal_cff" )
process.load( "Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff" )
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
from HLTrigger.Configuration.AutoCondGlobalTag import AutoCondGlobalTag
process.GlobalTag = AutoCondGlobalTag( process.GlobalTag, 'auto:com10' )

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
# for data:
# FIXME: very (too) simple to replace functionality from removed coreTools.py
process.patElectrons.addGenMatch  = False
process.patJets.addGenPartonMatch = False
process.patJets.addGenJetMatch    = False
process.patMETs.addGenMET         = False
process.patMuons.addGenMatch      = False
process.patPhotons.addGenMatch    = False
process.patTaus.addGenMatch       = False
process.patTaus.addGenJetMatch    = False
process.patJetCorrFactors.levels += [ 'L2L3Residual' ]
process.load( "PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff" )
