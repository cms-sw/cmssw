#ifndef TkGluedMeasurementDet_H
#define TkGluedMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"

class GluedGeomDet;
class SiStripRecHitMatcher;

class TkGluedMeasurementDet : public MeasurementDet {
public:

  TkGluedMeasurementDet( const GluedGeomDet* gdet,const SiStripRecHitMatcher* matcher,
			 const MeasurementDet* monoDet,
			 const MeasurementDet* stereoDet);

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&) const;

  const GluedGeomDet& specificGeomDet() const {return *theGeomDet;}

  virtual std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& startingState, 
		    const Propagator&, 
		    const MeasurementEstimator&) const;
  
private:

  const GluedGeomDet*         theGeomDet;
  const SiStripRecHitMatcher*       theMatcher;
  const TkStripMeasurementDet*       theMonoDet;
  const TkStripMeasurementDet*       theStereoDet;

  RecHitContainer 
  projectOnGluedDet( const RecHitContainer& hits,
		     const TrajectoryStateOnSurface& ts) const;

  void checkProjection(const TrajectoryStateOnSurface& ts, 
		       const RecHitContainer& monoHits, 
		       const RecHitContainer& stereoHits) const;
  void checkHitProjection(const TransientTrackingRecHit& hit,
			  const TrajectoryStateOnSurface& ts, 
			  const GeomDet& det) const;
};

#endif
