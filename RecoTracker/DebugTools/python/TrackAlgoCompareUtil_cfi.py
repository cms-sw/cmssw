import FWCore.ParameterSet.Config as cms

trackAlgoCompareUtil = cms.EDProducer("TrackAlgoCompareUtil",

    # recoTrack labels A and B
    trackLabel_algoA = cms.InputTag("generalTracks"),
    trackLabel_algoB = cms.InputTag("generalTracks"),

    # recoVertex labels A and B
    UseVertex = cms.bool(True),
    vertexLabel_algoA = cms.InputTag("offlinePrimaryVertices"),
    vertexLabel_algoB = cms.InputTag("offlinePrimaryVertices"),

    # trackingParticle/vertex labels (efficiency and fakes)
    trackingParticleLabel_effic = cms.InputTag("mix","MergedTrackTruth"),
    trackingParticleLabel_fakes = cms.InputTag("mix","MergedTrackTruth"),

    # beam spot label (used to calc. points of closest approach...etc.)
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),

    # associator label (by hits default)
    UseAssociators = cms.bool(True),
    assocLabel_algoA = cms.untracked.string('trackAssociatorByHits'),
    assocLabel_algoB = cms.untracked.string('trackAssociatorByHits'),
    
    # if an asociation map exist in the event, use it--its faster (set UseAssociatos = cms.bool(False)
    associatormap_algoA = cms.InputTag("trackingParticleRecoTrackAsssociation"),
    associatormap_algoB = cms.InputTag("trackingParticleRecoTrackAsssociation")
)



