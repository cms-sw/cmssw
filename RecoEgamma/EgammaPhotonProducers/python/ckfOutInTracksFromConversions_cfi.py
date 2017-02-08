import FWCore.ParameterSet.Config as cms

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFTrajectoryFitterForOutIn_cfi import *
#TrackProducers
ckfOutInTracksFromConversions = cms.EDProducer("TrackProducerWithSCAssociation",
    src = cms.InputTag("conversionTrackCandidates","outInTracksFromConversions"),
    recoTrackSCAssociationCollection = cms.string('outInTrackSCAssociationCollection'),
    producer = cms.string('conversionTrackCandidates'),
    Fitter = cms.string('KFFitterForOutIn'),
    trackCandidateSCAssociationCollection = cms.string('outInTrackCandidateSCAssociationCollection'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    #string AlgorithmName = "ecalSeededConv"
    AlgorithmName = cms.string('outInEcalSeededConv'),
    ComponentName = cms.string('ckfOutInTracksFromConversions'),
    #string Propagator = "PropagatorWithMaterial"
    Propagator = cms.string('alongMomElePropagator'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    MeasurementTracker = cms.string(''),
    MeasurementTrackerEvent = cms.InputTag('MeasurementTrackerEvent'),                   
                              
)


