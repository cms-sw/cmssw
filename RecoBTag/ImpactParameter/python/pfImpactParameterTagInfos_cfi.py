import FWCore.ParameterSet.Config as cms

pfImpactParameterTagInfos = cms.EDProducer("CandIPProducer",
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    computeProbabilities = cms.bool(True),
    ghostTrackPriorDeltaR = cms.double(0.03),
    jetDirectionUsingGhostTrack = cms.bool(False),
    minimumNumberOfPixelHits = cms.int32(2),
    jetDirectionUsingTracks = cms.bool(False),
    computeGhostTrack = cms.bool(True),
    useTrackQuality = cms.bool(False),
    maximumChiSquared = cms.double(5.0),
    #this is candidate specific
    jets = cms.InputTag("ak4PFJetsCHS"),
    candidates = cms.InputTag("particleFlow"),
    maxDeltaR= cms.double(0.4)
)
