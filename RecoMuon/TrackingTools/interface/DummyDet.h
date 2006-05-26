#ifndef TrackingTools_DummyDet_H
#define TrackingTools_DummyDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/MediumProperties.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
class Propagator;
class MeasurementEstimator;
class TrajectoryStateOnSurface;

typedef std::vector<GeomDetUnit*> DetUnitContainer;

/** Dummy Det without bounds. 
 *  Used in special RecHits (currently VertexRecHit).
 */
class DummyDet : public MeasurementDet {

public:
  
  DummyDet(const GeomDet * gdet, const BoundPlane* plane) : MeasurementDet(gdet), thePlane(plane) {}

  DummyDet(const BoundPlane* plane) : MeasurementDet(0), thePlane(plane) {}

  virtual ~DummyDet() {}
  // Methods of DummyDet

  void addRecHit(TransientTrackingRecHit* rhit) { rhits.push_back(rhit); }

  // Methods of Det

  virtual void clear() { rhits.clear(); }

  virtual RecHitContainer recHits() const { return rhits; }
  
  virtual const BoundSurface& surface() const { return *thePlane; }
  
  virtual DetUnitContainer detUnits() const { return DetUnitContainer(); }

  virtual std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& startingState, 
		    const Propagator&, 
		    const MeasurementEstimator&) const 
  { std::vector<TrajectoryMeasurement> tms; 
    return tms; 
   }

virtual RecHitContainer recHits( const TrajectoryStateOnSurface&) const {return rhits;}


private:

  std::vector<TrajectoryMeasurement> 
  tm(const TrajectoryStateOnSurface & tsos, 
     const MeasurementEstimator & aEstimator) const;

  ConstReferenceCountingPointer<BoundPlane> thePlane;
  RecHitContainer rhits;
  
};
#endif

