import FWCore.ParameterSet.Config as cms

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFFittingSmootherForInOut_cfi import *
softConversionIOTracks = cms.EDFilter("TrackProducerWithBCAssociation",
    src = cms.InputTag("softConversionTrackCandidates","softIOTrackCandidates"),
    recoTrackSCAssociationCollection = cms.string('inOutTrackClusterAssociationCollection'),
    producer = cms.string('softConversionTrackCandidates'),
    Fitter = cms.string('KFFittingSmootherForInOut'),
    useHitsSplitting = cms.bool(False),
    trackCandidateSCAssociationCollection = cms.string('inOutTrackCandidateClusterAssociationCollection'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    #string AlgorithmName = "ecalSeededConv"
    AlgorithmName = cms.string('undefAlgorithm'),
    ComponentName = cms.string('softConversionIOTracks'),
    #string Propagator = "PropagatorWithMaterial"
    Propagator = cms.string('alongMomElePropagator'),
    beamSpot = cms.InputTag("offlineBeamSpot")
)



