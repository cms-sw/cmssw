import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

## MessageLogger
process.load( "FWCore.MessageLogger.MessageLogger_cfi" )

## Options and Output Report
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool( False )
)

## Source
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
process.source = cms.Source(
  "PoolSource"
, fileNames = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion = 'CMSSW_12_5_0_pre3'
                        , globalTag    = 'phase1_2022_realistic'
                        )
  )
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 1000 ) # reduce number of events for testing
)

## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string( autoCond[ 'startup' ] )
process.load("Configuration.StandardSequences.MagneticField_cff")

## Standard PAT Configuration File
process.load("PhysicsTools.PatAlgos.patSequences_cff")
process.patJetCorrFactors.useRho = False

process.selectedPatMuons.cut = 'isTrackerMuon=1 & isGlobalMuon=1 & innerTrack.numberOfValidHits>=11 & globalTrack.normalizedChi2<10.0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & abs(dB)<0.02 & (trackIso+caloIso)/pt<0.05'

## ---
## Define the path
## ---
process.p = cms.Path(
  process.patDefaultSequence
)

### ========
### Plug-ins
### ========

## ---
## PAT trigger matching
## --
process.muonTriggerMatchHLTMuons = cms.EDProducer(
  # matching in DeltaR, sorting by best DeltaR
  "PATTriggerMatcherDRLessByR"
  # matcher input collections
, src     = cms.InputTag( 'cleanPatMuons' )
, matched = cms.InputTag( 'patTrigger' )
  # selections of trigger objects
, matchedCuts = cms.string( 'type( "TriggerMuon" ) && path( "HLT_Mu24_v*", 1, 0 )' ) # input does not yet have the 'saveTags' parameter in HLT
  # selection of matches
, maxDPtRel   = cms.double( 0.5 ) # no effect here
, maxDeltaR   = cms.double( 0.5 )
, maxDeltaEta = cms.double( 0.2 ) # no effect here
  # definition of matcher output
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)

### ============
### Python tools
### ============

## --
## Switch to selected PAT objects in the main work flow
## --
from PhysicsTools.PatAlgos.tools.coreTools import removeCleaning
removeCleaning( process, outputInProcess = False )

## --
## Switch on PAT trigger
## --
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTriggerMatching( process, triggerMatchers = [ 'muonTriggerMatchHLTMuons' ], outputModule = '' )
# Switch to selected PAT objects in the trigger matching
removeCleaningFromTriggerMatching( process, outputModule = '' )

## ---
## Add analysis
## ---
process.TFileService = cms.Service( "TFileService",
    fileName = cms.string( 'analyzePatTrigger_onTheFly.root' )
)
process.triggerAnalysis = cms.EDAnalyzer( "PatTriggerAnalyzer",
    trigger      = cms.InputTag( "patTrigger" ),
    triggerEvent = cms.InputTag( "patTriggerEvent" ),
    muons        = cms.InputTag( "selectedPatMuons" ),
    muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' ),
    minID = cms.uint32( 81 ),
    maxID = cms.uint32( 96 )
)
process.p += process.triggerAnalysis
