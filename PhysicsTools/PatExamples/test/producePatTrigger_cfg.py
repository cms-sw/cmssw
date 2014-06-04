
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
process.maxEvents.input     = 1000 # reduce number of events for testing.
process.options.wantSummary = False # to suppress the long output at the end of the job
# specific
process.patJetCorrFactors.useRho = False
process.patJets.addTagInfos      = False # to save space
process.selectedPatMuons.cut     = 'isTrackerMuon=1 & isGlobalMuon=1 & innerTrack.numberOfValidHits>=11 & globalTrack.normalizedChi2<10.0  & globalTrack.hitPattern().numberOfValidMuonHits(\'TRACK_HITS\') > 0 & abs(dB)<0.02 & (trackIso+caloIso)/pt<0.05'

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
