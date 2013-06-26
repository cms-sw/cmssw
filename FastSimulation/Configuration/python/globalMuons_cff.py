import FWCore.ParameterSet.Config as cms

from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *
from RecoMuon.GlobalMuonProducer.globalMuons_cfi import *

Chi2EstimatorForMuRefit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForMuRefit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

GlbMuKFFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('GlbMuKFFitter'),
    Estimator = cms.string('Chi2EstimatorForMuRefit'),
    Propagator = cms.string('SmartPropagatorAny'),
    Updator = cms.string('KFUpdator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    minHits = cms.int32(3)
)


