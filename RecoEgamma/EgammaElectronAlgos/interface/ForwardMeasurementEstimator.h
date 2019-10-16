#ifndef ForwardMeasurementEstimator_H
#define ForwardMeasurementEstimator_H

// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ForwardMeasurementEstimator
//
/**\class ForwardMeasurementEstimator EgammaElectronAlgos/ForwardMeasurementEstimator

 Description: Class defining the search area in the forward disks in the pixel match, ported from ORCA

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include <utility>

class ForwardMeasurementEstimator {
public:
  ForwardMeasurementEstimator(float phiMin, float phiMax, float rMin, float rMax)
      : thePhiMin(phiMin), thePhiMax(phiMax), theRMin(rMin), theRMax(rMax) {}

  void setPhiRange(float dummyphiMin, float dummyphiMax) {
    thePhiMin = dummyphiMin;
    thePhiMax = dummyphiMax;
  }
  void setRRange(float rmin, float rmax) {
    theRMin = rmin;
    theRMax = rmax;
  }
  void setRRangeI(float rmin, float rmax) {
    theRMinI = rmin;
    theRMaxI = rmax;
  }

  bool estimate(const GlobalPoint& vprim,
                const TrajectoryStateOnSurface& ts,
                const GlobalPoint& gp) const;

private:
  float thePhiMin;
  float thePhiMax;
  float theRMin;
  float theRMax;
  float theRMinI;
  float theRMaxI;
};

#endif  // ForwardMeasurementEstimator_H
