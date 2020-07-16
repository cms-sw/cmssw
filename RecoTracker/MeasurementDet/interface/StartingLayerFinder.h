#ifndef TkNavigation_StartingLayerFinder_H_
#define TkNavigation_StartingLayerFinder_H_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include <vector>

/** Finds the nearest navigable layer.
 *  Needed to start trajectory building in case the seed does not 
 *  have a DetLayer
 */

class StartingLayerFinder {
public:
  StartingLayerFinder(Propagator const& aPropagator, MeasurementTracker const& tracker)
      : thePropagator(aPropagator),
        theMeasurementTracker(tracker),
        theFirstNegPixelFwdLayer(0),
        theFirstPosPixelFwdLayer(0) {}

  std::vector<const DetLayer*> operator()(const FreeTrajectoryState& aFts, float dr, float dz) const;

private:
  const BarrelDetLayer* firstPixelBarrelLayer() const;
  const std::vector<const ForwardDetLayer*> firstNegPixelFwdLayer() const;
  const std::vector<const ForwardDetLayer*> firstPosPixelFwdLayer() const;

  Propagator const& thePropagator;
  MeasurementTracker const& theMeasurementTracker;
  mutable bool thePixelLayersValid = false;
  mutable const BarrelDetLayer* theFirstPixelBarrelLayer = nullptr;
  mutable std::vector<const ForwardDetLayer*> theFirstNegPixelFwdLayer;
  mutable std::vector<const ForwardDetLayer*> theFirstPosPixelFwdLayer;

  void checkPixelLayers() const;
};

#endif
