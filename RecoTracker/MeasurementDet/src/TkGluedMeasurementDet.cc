#include "RecoTracker/MeasurementDet/interface/TkGluedMeasurementDet.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoTracker/MeasurementDet/interface/NonPropagatingDetMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

TkGluedMeasurementDet::TkGluedMeasurementDet( const GluedGeomDet* gdet, 
					      const SiStripRecHitMatcher* matcher,
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
  LocalVector tkDir = (ts.isValid() ? ts.localDirection() : surface().toLocal( position()-GlobalPoint(0,0,0)));

  if (monoHits.empty()) return stereoHits;
  else if (stereoHits.empty()) return monoHits;
  else {    
    // convert stereo hits to type expected by matcher
    SiStripRecHitMatcher::SimpleHitCollection  vsStereoHits;
    for (RecHitContainer::const_iterator stereoHit = stereoHits.begin();
	 stereoHit != stereoHits.end(); stereoHit++) {
      const TrackingRecHit* tkhit = (**stereoHit).hit();
      const SiStripRecHit2D* verySpecificStereoHit =
	dynamic_cast<const SiStripRecHit2D*>(tkhit);
      if (verySpecificStereoHit == 0) {
	throw MeasurementDetException("TkGluedMeasurementDet ERROR: stereoHit is not SiStripRecHit2D");
      }
      vsStereoHits.push_back( verySpecificStereoHit);
    }

    // convert mono hits to type expected by matcher
    for (RecHitContainer::const_iterator monoHit = monoHits.begin();
	 monoHit != monoHits.end(); monoHit++) {
      const TrackingRecHit* tkhit = (**monoHit).hit();
      const SiStripRecHit2D* verySpecificMonoHit =
	dynamic_cast<const SiStripRecHit2D*>(tkhit);
      if (verySpecificMonoHit == 0) {
	throw MeasurementDetException("TkGluedMeasurementDet ERROR: monoHit is not SiStripRecHit2D");
      }

      edm::OwnVector<SiStripMatchedRecHit2D> tmp =
	theMatcher->match( verySpecificMonoHit, vsStereoHits.begin(), vsStereoHits.end(),
			  &specificGeomDet(), tkDir);

      if(tmp.size()){
	for (edm::OwnVector<SiStripMatchedRecHit2D>::const_iterator i=tmp.begin();
	     i != tmp.end(); i++) {
	  result.push_back( TSiStripMatchedRecHit::build( &geomDet(), &(*i)));
	}
      }else{
	LogDebug("MeasurementDet") << "in TkGluedMeasurementDet, no stereo hit matched with the mono one" ;
	result.push_back(*monoHit);
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
