import FWCore.ParameterSet.Config as cms

# magnetic field
# geometry
# from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
# from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *
from RecoMuon.GlobalMuonProducer.globalMuons_cfi import *
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
Chi2EstimatorForMuRefit = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone()
Chi2EstimatorForMuRefit.ComponentName = cms.string('Chi2EstimatorForMuRefit')
Chi2EstimatorForMuRefit.nSigma = 3.0
Chi2EstimatorForMuRefit.MaxChi2 = 100000.0


from TrackingTools.TrackFitters.TrackFitters_cff import *
GlbMuKFFitter = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = cms.string('GlbMuKFFitter'),
    Estimator = cms.string('Chi2EstimatorForMuRefit'),
    Propagator = cms.string('SmartPropagatorAnyRK'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
# FastSim doesn't use Runge Kute for propagation
fastSim.toModify(GlbMuKFFitter,
                 Propagator = "SmartPropagatorAny")


