#ifndef TrajectoryAtInvalidHit_H
#define TrajectoryAtInvalidHit_H

// Class to hold the trajectory information at an invalid hit
// For matched layers, the invalid hit on the trajectory is located
// on the matched surface. To compare with rechits propagate the 
// information to the actual sensor surface for rphi or stereo 

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

class Topology;
class TransientTrackingRecHit;
class StripTopology;
class PixelTopology;
class TrajectoryAtInvalidHit {
public:

  TrajectoryAtInvalidHit( const TrajectoryMeasurement&, 
			const TrackerGeometry * tracker, 
			const Propagator& propagator,
			const uint mono = 0);

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

  uint monodet_id() const;
  bool withinAcceptance() const;

  bool isDoubleSided(uint iidd) const;
  TrajectoryStateOnSurface tsos() const;

private:

  TrajectoryStateOnSurface theCombinedPredictedState;
  float locX,locY, locZ;
  float locXError, locYError;
  float locDxDz, locDyDz;
  float globX, globY, globZ;
  uint iidd;
  bool acceptance;

  ConstReferenceCountingPointer<TransientTrackingRecHit> theHit;
};

#endif
