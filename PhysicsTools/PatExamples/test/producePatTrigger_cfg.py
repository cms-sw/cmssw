# Start with pre-defined skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# Switch to selected PAT objects
from PhysicsTools.PatAlgos.tools.coreTools import removeCleaning
removeCleaning( process )

# Define the path
process.p = cms.Path(
    process.patDefaultSequence
)

process.maxEvents.input     = -1 # Reduce number of events for testing.
process.out.fileName        = 'patTuple.root'
process.options.wantSummary = False # to suppress the long output at the end of the job

process.selectedPatMuons.cut = 'isTrackerMuon=1 & isGlobalMuon=1 & innerTrack.numberOfValidHits>=11 & globalTrack.normalizedChi2<10.0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & abs(dB)<0.02 & (trackIso+caloIso)/pt<0.05'

# PAT trigger
process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff" )
process.muonTriggerMatchHLTMuons = cms.EDProducer( "PATTriggerMatcherDRLessByR",
    src     = cms.InputTag( "selectedPatMuons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( 'TriggerMuon' ),
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_Mu9' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( True )
)
process.patTriggerMatcher += process.muonTriggerMatchHLTMuons
process.patTriggerMatcher.remove( process.patTriggerMatcherElectron )
process.patTriggerMatcher.remove( process.patTriggerMatcherMuon )
process.patTriggerMatcher.remove( process.patTriggerMatcherTau )
process.patTriggerEvent.patTriggerMatches = [ "muonTriggerMatchHLTMuons" ]
from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
switchOnTrigger( process )
