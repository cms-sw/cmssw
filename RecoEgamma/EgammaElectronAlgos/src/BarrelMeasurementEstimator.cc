// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      BarrelMeasurementEstimator
//
/**\class ElectronSeedProducer EgammaElectronAlgos/BarrelMeasurementEstimator

 Description: MeasurementEstimator for Pixel Barrel, ported from ORCA

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

bool BarrelMeasurementEstimator::estimate(const GlobalPoint& vprim,
                                          const TrajectoryStateOnSurface& absolute_ts,
                                          const GlobalPoint& absolute_gp) const {
  GlobalVector ts = absolute_ts.globalParameters().position() - vprim;
  GlobalVector gp = absolute_gp - vprim;

  float myZ = gp.z();
  float zDiff = myZ - ts.z();
  float myZmax = theZMax;
  float myZmin = theZMin;
  if ((std::abs(myZ) < 30.f) & (gp.perp() > 8.f)) {
    myZmax = 0.09f;
    myZmin = -0.09f;
  }

  if ((zDiff >= myZmax) | (zDiff <= myZmin))
      return false;

  float phiDiff = normalizedPhi(gp.barePhi() - ts.barePhi());

  return (phiDiff < thePhiMax) & (phiDiff > thePhiMin);
}
