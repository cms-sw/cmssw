import FWCore.ParameterSet.Config as cms

hltDeepBLifetimeTagInfosPFPuppi = cms.EDProducer("CandIPProducer",
    candidates = cms.InputTag("particleFlowTmp"),
    computeGhostTrack = cms.bool(True),
    computeProbabilities = cms.bool(True),
    ghostTrackPriorDeltaR = cms.double(0.03),
    jetDirectionUsingGhostTrack = cms.bool(False),
    jetDirectionUsingTracks = cms.bool(False),
    jets = cms.InputTag("hltAK4PFPuppiJets"),
    maxDeltaR = cms.double(0.4),
    maximumChiSquared = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(3),
    minimumNumberOfPixelHits = cms.int32(2),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    useTrackQuality = cms.bool(False)
)
