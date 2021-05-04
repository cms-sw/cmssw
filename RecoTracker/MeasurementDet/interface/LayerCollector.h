#ifndef TkNavigation_LayerCollector_H_
#define TkNavigation_LayerCollector_H_
/**
 *   \class LayerCollector
 *   Class collecting all layers of the tracker.  
 *   
 *   
 */

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/MeasurementDet/interface/StartingLayerFinder.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

class NavigationSchool;

class LayerCollector {
private:
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;
  typedef std::pair<float, float> Range;

public:
  LayerCollector(NavigationSchool const* aSchool,
                 const Propagator* aPropagator,
                 const MeasurementTracker* tracker,
                 float dr,
                 float dz)
      : theSchool(aSchool),
        thePropagator(aPropagator),
        theStartingLayerFinder{*aPropagator, *tracker},
        theDeltaR(dr),
        theDeltaZ(dz) {}

  std::vector<const DetLayer*> allLayers(const FTS& aFts) const;
  std::vector<const BarrelDetLayer*> barrelLayers(const FTS& aFts) const;
  std::vector<const ForwardDetLayer*> forwardLayers(const FTS& aFts) const;

  const Propagator* propagator() const { return thePropagator; }
  float deltaR() const { return theDeltaR; }
  float deltaZ() const { return theDeltaZ; }

private:
  NavigationSchool const* theSchool;
  const Propagator* thePropagator;
  const StartingLayerFinder theStartingLayerFinder;
  float theDeltaR;
  float theDeltaZ;

  inline bool rangesIntersect(const Range& a, const Range& b) const {
    if (a.first > b.second || b.first > a.second)
      return false;
    else
      return true;
  }
};

#endif  //TR_LayerCollector_H_
