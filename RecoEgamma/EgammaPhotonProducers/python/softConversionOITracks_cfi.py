import FWCore.ParameterSet.Config as cms

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFTrajectoryFitterForOutIn_cfi import *
#TrackProducers
softConversionOITracks = cms.EDFilter("TrackProducerWithBCAssociation",
    src = cms.InputTag("softConversionTrackCandidates","softOITrackCandidates"),
    recoTrackSCAssociationCollection = cms.string('outInTrackClusterAssociationCollection'),
    producer = cms.string('softConversionTrackCandidates'),
    Fitter = cms.string('KFFitterForOutIn'),
    useHitsSplitting = cms.bool(False),
    trackCandidateSCAssociationCollection = cms.string('outInTrackCandidateClusterAssociationCollection'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    ComponentName = cms.string('softConversionOITracks'),
    #string Propagator = "PropagatorWithMaterial"
    Propagator = cms.string('alongMomElePropagator'),
    beamSpot = cms.InputTag("offlineBeamSpot")
)



