#ifndef TkGluedMeasurementDet_H
#define TkGluedMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"

class GluedGeomDet;
class SiStripRecHitMatcher;

class TkGluedMeasurementDet : public MeasurementDet {
public:

  TkGluedMeasurementDet( const GluedGeomDet* gdet, SiStripRecHitMatcher* matcher,
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
  SiStripRecHitMatcher*       theMatcher;
  const MeasurementDet*       theMonoDet;
  const MeasurementDet*       theStereoDet;

};

#endif
