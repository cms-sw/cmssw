import FWCore.ParameterSet.Config as cms

process = cms.Process( "PAT" )

## Options and Output Report
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool( True )
)

## Source
process.source = cms.Source( "PoolSource"
, fileNames = cms.untracked.vstring(
    '/store/data/Run2011A/HT/RAW/v1/000/166/565/DE60076D-A290-E011-9BC9-003048F118C6.root'
  )
, lumisToProcess = cms.untracked.VLuminosityBlockRange(
    '166565:12-166565:12'
  )
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 100 )
)

## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load( "Configuration.StandardSequences.Services_cff" )
process.load( "Configuration.StandardSequences.Geometry_cff" )
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag( process.GlobalTag, 'auto:com10' )
process.load( "Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff" )

# RAW to DIGI and RECO pre-requisites
process.load( "Configuration.StandardSequences.RawToDigi_Data_cff" )
process.load( "L1Trigger.Configuration.L1Reco_cff" )
process.p = cms.Path(
  process.gctDigis
* process.gtDigis
* process.gtEvmDigis
* process.scalersRawToDigi
* process.L1Reco
)

## Output
from L1Trigger.Configuration.L1Trigger_EventContent_cff import L1TriggerAOD
process.out = cms.OutputModule(
  "PoolOutputModule"
, fileName       = cms.untracked.string( 'patTrigger_dataFromRAW.root' )
, SelectEvents   = cms.untracked.PSet(
    SelectEvents = cms.vstring(
      'p'
    )
  )
, outputCommands = cms.untracked.vstring(
    'drop *'
  )
)
process.outpath = cms.EndPath(
  process.out
)

## PAT trigger
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTrigger( process, sequence = 'p', hltProcess = '*' ) # to update event content
