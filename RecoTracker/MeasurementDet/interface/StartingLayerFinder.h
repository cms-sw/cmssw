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

class PTrajectoryStateOnDet;

class StartingLayerFinder {

private:

  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;
  typedef std::pair<float, float> Range;

public: 

  StartingLayerFinder(const Propagator* aPropagator, const MeasurementTracker*  tracker ) :

    thePropagator(aPropagator),
    theMeasurementTracker(tracker),
    thePixelLayersValid(false),
    theFirstPixelBarrelLayer(nullptr),
    theFirstNegPixelFwdLayer(0),
    theFirstPosPixelFwdLayer(0) { }

  ~StartingLayerFinder() {}

  std::vector<const DetLayer*> startingLayers(const FTS& aFts, float dr, float dz) const;
  

  std::vector<const DetLayer*> startingLayers(const TrajectorySeed& aSeed) const;

  const BarrelDetLayer* firstPixelBarrelLayer() const;
  const std::vector<const ForwardDetLayer*> firstNegPixelFwdLayer() const;
  const std::vector<const ForwardDetLayer*> firstPosPixelFwdLayer() const;

  const Propagator* propagator() const {return thePropagator;}



private:

  const Propagator* thePropagator;
  const MeasurementTracker*     theMeasurementTracker;
  mutable bool thePixelLayersValid;
  mutable const BarrelDetLayer* theFirstPixelBarrelLayer;
  mutable std::vector<const ForwardDetLayer*> theFirstNegPixelFwdLayer;
  mutable std::vector<const ForwardDetLayer*> theFirstPosPixelFwdLayer;


  void checkPixelLayers() const;
  
  
  
  inline bool rangesIntersect( const Range& a, const Range& b) const {
    if ( a.first > b.second || b.first > a.second) return false;
    else return true;
  }




};
#endif //TR_StartingLayerFinder_H_
