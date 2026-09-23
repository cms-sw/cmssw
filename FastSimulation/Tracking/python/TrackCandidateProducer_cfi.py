import FWCore.ParameterSet.Config as cms

trackCandidateProducer = cms.EDProducer(
    "TrackCandidateProducer",
    recHitCombinations = cms.InputTag("fastMatchedTrackerRecHitCombinations"),
    # The smallest number of crossed layers to make a candidate
    MinNumberOfCrossedLayers = cms.uint32(5),

    src = cms.InputTag("tobTecStepSeeds"),

    OverlapCleaning = cms.bool(False),

    # Split matched hits? 
    SplitHits = cms.bool(True),
    simTracks = cms.InputTag('fastSimProducer'),
    
    propagator = cms.string('PropagatorWithMaterialOpposite')
)


from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(trackCandidateProducer, recHitCombinations = cms.InputTag("fastTrackerRecHitCombinations"))
