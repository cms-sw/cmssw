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
    pickRelValInputFiles()
  )
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 1000 ) # reduce number of events for testing
)

## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START38_V14::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

## Standard PAT Configuration File
process.load("PhysicsTools.PatAlgos.patSequences_cff")
# process.load("PhysicsTools.PatAlgos.patTestJEC_cfi")

process.patJets.addTagInfos  = False # to save space
process.selectedPatMuons.cut = 'isTrackerMuon=1 & isGlobalMuon=1 & innerTrack.numberOfValidHits>=11 & globalTrack.normalizedChi2<10.0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & abs(dB)<0.02 & (trackIso+caloIso)/pt<0.05'

## ---
## Define the path
## ---
process.p = cms.Path(
  process.patDefaultSequence
)

## ---
## PAT trigger matching
## --
process.muonTriggerMatchHLTMuons = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"
, src     = cms.InputTag( 'cleanPatMuons' )
, matched = cms.InputTag( 'patTrigger' )
, andOr          = cms.bool( False )
, filterIdsEnum  = cms.vstring( 'TriggerMuon' )
, filterIds      = cms.vint32( 0 )
, filterLabels   = cms.vstring( '*' )
, pathNames      = cms.vstring( 'HLT_Mu9' )
, collectionTags = cms.vstring( '*' )
, maxDPtRel   = cms.double( 0.5 ) # no effect here
, maxDeltaR   = cms.double( 0.5 )
, maxDeltaEta = cms.double( 0.2 ) # no effect here
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)

### ============
### Python tools
### ============
### Attention: order matters!

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
