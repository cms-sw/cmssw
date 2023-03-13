import FWCore.ParameterSet.Config as cms

from RecoTracker.PixelLowPtUtilities.clusterShapeTrackFilter_cfi import clusterShapeTrackFilter as _clusterShapeTrackFilter
clusterFilter = _clusterShapeTrackFilter.clone(
    ptMin = 1.5,
)

from RecoHI.HiTracking.hiPixelTrackFilter_cfi import hiPixelTrackFilter as _hiPixelTrackFilter
hiFilter = _hiPixelTrackFilter.clone()

from RecoTracker.PixelTrackFitting.pixelTrackFilterByKinematics_cfi import pixelTrackFilterByKinematics as _pixelTrackFilterByKinematics
kinematicFilter = _pixelTrackFilterByKinematics.clone(
    ptMin = 0.7,
)

from RecoHI.HiTracking.hiProtoTrackFilter_cfi import hiProtoTrackFilter as _hiProtoTrackFilter
hiProtoTrackFilter = _hiProtoTrackFilter.clone()

hiConformalPixelFilter = _hiPixelTrackFilter.clone(
    ptMin = 0.25 ,
    chi2 = 80.0,
    nSigmaTipMaxTolerance = 999.0,
    tipMax = 999.0,
    nSigmaLipMaxTolerance = 14.0,
    lipMax = 999.0,
)
