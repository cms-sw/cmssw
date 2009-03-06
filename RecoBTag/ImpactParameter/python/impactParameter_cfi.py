import FWCore.ParameterSet.Config as cms

impactParameterTagInfos = cms.EDProducer("TrackIPProducer",
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    maximumDecayLength = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    jetTracks = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
    minimumNumberOfPixelHits = cms.int32(2),
    jetDirectionUsingTracks = cms.bool(False),
    computeProbabilities = cms.bool(True),
    maximumDistanceToJetAxis = cms.double(0.07),
    maximumChiSquared = cms.double(5.0)
)


