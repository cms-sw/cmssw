import FWCore.ParameterSet.Config as cms

#ConeIsolation
coneIsolationTauJetTags = cms.EDProducer("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(0.0),
    MaximumTransverseImpactParameter = cms.double(0.03),
    VariableConeParameter = cms.double(3.5),
    useVertex = cms.bool(True),
    MinimumNumberOfHits = cms.int32(8),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("offlinePrimaryVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(False),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("offlineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    MaximumChiSquared = cms.double(100.0)
)


