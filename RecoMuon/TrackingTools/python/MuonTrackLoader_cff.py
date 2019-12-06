import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.GeomPropagators.SmartPropagator_cff import *
from RecoMuon.TrackingTools.MuonUpdatorAtVertex_cff import *
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
Chi2EstimatorForMuonTrackLoader = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone()
Chi2EstimatorForMuonTrackLoader.ComponentName = cms.string('Chi2EstimatorForMuonTrackLoader')
Chi2EstimatorForMuonTrackLoader.nSigma = 3.0
Chi2EstimatorForMuonTrackLoader.MaxChi2 = 100000.0

import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
KFSmootherForMuonTrackLoader = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForMuonTrackLoader'),
    Estimator = cms.string('Chi2EstimatorForMuonTrackLoader'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagatorAnyRK')
)
from Configuration.Eras.Modifier_fastSim_cff import fastSim
# FastSim doesn't use Runge Kute for propagation
fastSim.toModify(KFSmootherForMuonTrackLoader,
                 Propagator = "SmartPropagatorAny")

KFSmootherForMuonTrackLoaderL3 = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForMuonTrackLoaderL3'),
    Estimator = cms.string('Chi2EstimatorForMuonTrackLoader'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagatorAnyOpposite')
)

MuonTrackLoaderForSTA = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        MuonUpdatorAtVertex,
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        DoSmoothing = cms.bool(False),
        VertexConstraint = cms.bool(True),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        TTRHBuilder = cms.string('WithAngleAndTemplate')
    )
)
MuonTrackLoaderForGLB = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        MuonUpdatorAtVertex,
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        DoSmoothing = cms.bool(True),
        VertexConstraint = cms.bool(False),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        TTRHBuilder = cms.string('WithAngleAndTemplate')
    )
)
MuonTrackLoaderForL2 = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        MuonUpdatorAtVertex,
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        DoSmoothing = cms.bool(False),
        VertexConstraint = cms.bool(True),
        beamSpot = cms.InputTag("hltOfflineBeamSpot"),
        TTRHBuilder = cms.string('WithAngleAndTemplate')
    )
)
MuonTrackLoaderForL3 = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        MuonUpdatorAtVertex,
        PutTkTrackIntoEvent = cms.untracked.bool(True),
        Smoother = cms.string('KFSmootherForMuonTrackLoaderL3'),
        SmoothTkTrack = cms.untracked.bool(False),
        MuonSeededTracksInstance = cms.untracked.string('L2Seeded'),
        VertexConstraint = cms.bool(False),
        DoSmoothing = cms.bool(True),
        beamSpot = cms.InputTag("hltOfflineBeamSpot"),
        TTRHBuilder = cms.string('WithAngleAndTemplate')
    )
)
MuonTrackLoaderForCosmic = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        MuonUpdatorAtVertexAnyDirection,
        PutTrajectoryIntoEvent = cms.untracked.bool(False),
        VertexConstraint = cms.bool(False),
        AllowNoVertex = cms.untracked.bool(True),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        DoSmoothing = cms.bool(False),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        TTRHBuilder = cms.string('WithAngleAndTemplate')
    )
)

