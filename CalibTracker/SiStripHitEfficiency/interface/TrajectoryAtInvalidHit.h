#ifndef TrajectoryAtInvalidHit_H
#define TrajectoryAtInvalidHit_H

// Class to hold the trajectory information at a possibly invalid hit
// For matched layers, the invalid hit on the trajectory is located
// on the matched surface. To compare with rechits propagate the 
// information to the actual sensor surface for rphi or stereo 

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

class Topology;
class TrackingRecHit;
class StripTopology;
class PixelTopology;
class TrackerTopology;

class TrajectoryAtInvalidHit {
public:

  TrajectoryAtInvalidHit( const TrajectoryMeasurement&, 
			const TrackerTopology * tTopo, 
			const TrackerGeometry * tracker, 
			const Propagator& propagator,
			const unsigned int mono = 0);

  double localX() const;
  double localY() const;
  double localErrorX() const;
  double localErrorY() const;

  double localDxDz() const;
  double localDyDz() const;

  double localZ() const;

  double globalX() const;
  double globalY() const;
  double globalZ() const;

  unsigned int monodet_id() const;
  bool withinAcceptance() const;
  bool validHit() const;

  bool isDoubleSided(unsigned int iidd, const TrackerTopology* tTopo) const;
  TrajectoryStateOnSurface tsos() const;

private:

  TrajectoryStateOnSurface theCombinedPredictedState;
  float locX,locY, locZ;
  float locXError, locYError;
  float locDxDz, locDyDz;
  float globX, globY, globZ;
  unsigned int iidd;
  bool acceptance;
  bool hasValidHit;

  TrackingRecHit::ConstRecHitPointer theHit;
};

#endif
