// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ForwardMeasurementEstimator
//
/**\class ForwardMeasurementEstimator EgammaElectronAlgos/ForwardMeasurementEstimator

 Description: MeasurementEstimator for Pixel Endcap, ported from ORCA

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

bool ForwardMeasurementEstimator::estimate(const GlobalPoint& vprim,
                                           const TrajectoryStateOnSurface& absolute_ts,
                                           const GlobalPoint& absolute_gp) const {
  GlobalVector ts = absolute_ts.globalParameters().position() - vprim;
  GlobalVector gp = absolute_gp - vprim;

  float rDiff = gp.perp() - ts.perp();
  float rMin = theRMin;
  float rMax = theRMax;
  float myZ = gp.z();
  if ((std::abs(myZ) > 70.f) & (std::abs(myZ) < 170.f)) {
    rMin = theRMinI;
    rMax = theRMaxI;
  }

  if ((rDiff >= rMax) | (rDiff <= rMin))
    return false;

  float phiDiff = normalizedPhi(gp.barePhi() - ts.barePhi());

  return (phiDiff < thePhiMax) & (phiDiff > thePhiMin);
}
