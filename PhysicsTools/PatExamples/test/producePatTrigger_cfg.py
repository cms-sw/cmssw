
### ========
### Skeleton
### ========

## ---
## Start with pre-defined skeleton process
## ---
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## ---
## Modifications
## ---
# general
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
process.source.fileNames = pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_3_6'
                                               , relVal        = 'RelValProdTTbar'
                                               , globalTag     = 'START53_V14'
                                               , dataTier      = 'AODSIM'
                                               , maxVersions   = 2
                                               , numberOfFiles = -1
                                               )
process.maxEvents.input     = -1 # reduce number of events for testing.
process.options.wantSummary = False # to suppress the long output at the end of the job
# specific
process.selectedPatMuons.cut = 'isGlobalMuon && pt > 20. && abs(eta) < 2.1 && globalTrack.normalizedChi2 < 10. && track.hitPattern.trackerLayersWithMeasurement > 5 && globalTrack.hitPattern.numberOfValidMuonHits > 0 && abs(dB) < 0.2 && innerTrack.hitPattern.numberOfValidPixelHits > 0 && numberOfMatchedStations > 1 && (trackIso+caloIso)/pt<0.2'

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
, matchedCuts = cms.string( 'type( "TriggerMuon" ) && path( "HLT_IsoMu24_eta2p1_v*" )' )
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
removeCleaning( process )
# to save a bit of disk space
process.out.outputCommands += [ 'drop recoBaseTagInfosOwned_*_*_*'
                              , 'drop CaloTowers_*_*_*'
                              , 'drop recoGenJets_*_*_*'
                              ]

## --
## Switch on PAT trigger
## --
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTrigger( process ) # This is optional and can be omitted.
switchOnTriggerMatching( process, triggerMatchers = [ 'muonTriggerMatchHLTMuons' ] )
# Switch to selected PAT objects in the trigger matching
removeCleaningFromTriggerMatching( process )
