import FWCore.ParameterSet.Config as cms

# magnetic field
# geometry
# from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
# from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *
from RecoMuon.GlobalMuonProducer.globalMuons_cfi import *
Chi2EstimatorForMuRefit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForMuRefit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)


from TrackingTools.TrackFitters.TrackFitters_cff import *
GlbMuKFFitter = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone(
    ComponentName = cms.string('GlbMuKFFitter'),
    Estimator = cms.string('Chi2EstimatorForMuRefit'),
    Propagator = cms.string('SmartPropagatorAnyRK'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)



