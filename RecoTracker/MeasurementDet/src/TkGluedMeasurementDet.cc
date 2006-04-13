#include "RecoTracker/MeasurementDet/interface/TkGluedMeasurementDet.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoTracker/MeasurementDet/interface/NonPropagatingDetMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

TkGluedMeasurementDet::TkGluedMeasurementDet( const GluedGeomDet* gdet, 
					      SiStripRecHitMatcher* matcher,
					      const MeasurementDet* monoDet,
					      const MeasurementDet* stereoDet) :
  MeasurementDet(gdet), theGeomDet(gdet), 
  theMatcher(matcher), 
  theMonoDet( monoDet), theStereoDet(stereoDet)
{
  
}

TkGluedMeasurementDet::RecHitContainer 
TkGluedMeasurementDet::recHits( const TrajectoryStateOnSurface& ts) const
{
  RecHitContainer result;

  RecHitContainer monoHits = theMonoDet->recHits( ts);
  RecHitContainer stereoHits = theStereoDet->recHits( ts);

  if (monoHits.empty()) return stereoHits;
  else if (stereoHits.empty()) return monoHits;
  else {

    LocalVector tkDir = (ts.isValid() ? ts.localDirection() : surface().toLocal( position()-GlobalPoint(0,0,0)));

    // convert stereo hits to type expected by matcher
    SiStripRecHitMatcher::SimpleHitCollection  vsStereoHits;
    for (RecHitContainer::const_iterator stereoHit = stereoHits.begin();
	 stereoHit != stereoHits.end(); stereoHit++) {
      const TrackingRecHit* tkhit = (**stereoHit).hit();
      const SiStripRecHit2DLocalPos* verySpecificStereoHit =
	dynamic_cast<const SiStripRecHit2DLocalPos*>(tkhit);
      if (verySpecificStereoHit == 0) {
	throw MeasurementDetException("TkGluedMeasurementDet ERROR: stereoHit is not SiStripRecHit2DLocalPos");
      }
      vsStereoHits.push_back( verySpecificStereoHit);
    }

    // convert mono hits to type expected by matcher
    for (RecHitContainer::const_iterator monoHit = monoHits.begin();
	 monoHit != monoHits.end(); monoHit++) {
      const TrackingRecHit* tkhit = (**monoHit).hit();
      const SiStripRecHit2DLocalPos* verySpecificMonoHit =
	dynamic_cast<const SiStripRecHit2DLocalPos*>(tkhit);
      if (verySpecificMonoHit == 0) {
	throw MeasurementDetException("TkGluedMeasurementDet ERROR: monoHit is not SiStripRecHit2DLocalPos");
      }

      edm::OwnVector<SiStripRecHit2DMatchedLocalPos> tmp =
	theMatcher->match( verySpecificMonoHit, vsStereoHits.begin(), vsStereoHits.end(),
			  &specificGeomDet(), tkDir);

      for (edm::OwnVector<SiStripRecHit2DMatchedLocalPos>::const_iterator i=tmp.begin();
	   i != tmp.end(); i++) {
	result.push_back( new TSiStripMatchedRecHit( &geomDet(), &(*i)));
      }
    }
  }
  return result;
}

std::vector<TrajectoryMeasurement> 
TkGluedMeasurementDet::fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
					 const TrajectoryStateOnSurface& startingState, 
					 const Propagator&, 
					 const MeasurementEstimator& est) const
{
  NonPropagatingDetMeasurements realOne;
  return realOne.get( *this, stateOnThisDet, est);

}
