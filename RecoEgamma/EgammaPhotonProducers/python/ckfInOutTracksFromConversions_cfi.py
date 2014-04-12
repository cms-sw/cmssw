import FWCore.ParameterSet.Config as cms

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFFittingSmootherForInOut_cfi import *
ckfInOutTracksFromConversions = cms.EDProducer("TrackProducerWithSCAssociation",
    src = cms.InputTag("conversionTrackCandidates","inOutTracksFromConversions"),
    recoTrackSCAssociationCollection = cms.string('inOutTrackSCAssociationCollection'),
    producer = cms.string('conversionTrackCandidates'),
    Fitter = cms.string('KFFittingSmootherForInOut'),
    trackCandidateSCAssociationCollection = cms.string('inOutTrackCandidateSCAssociationCollection'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    #string AlgorithmName = "ecalSeededConv"
    AlgorithmName = cms.string('inOutEcalSeededConv'),
    ComponentName = cms.string('ckfInOutTracksFromConversions'),
    #string Propagator = "PropagatorWithMaterial"
    Propagator = cms.string('alongMomElePropagator'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    MeasurementTracker = cms.string(''),                              
    MeasurementTrackerEvent = cms.InputTag('MeasurementTrackerEvent'),                   
)


