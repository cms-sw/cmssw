#include "RecoTracker/MeasurementDet/interface/TkGluedMeasurementDet.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoTracker/MeasurementDet/interface/NonPropagatingDetMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "RecoTracker/MeasurementDet/interface/RecHitPropagator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

#include <typeinfo>
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorForTrackerHits.h"

using namespace std;

TkGluedMeasurementDet::TkGluedMeasurementDet( const GluedGeomDet* gdet, 
					      const SiStripRecHitMatcher* matcher,
					      const MeasurementDet* monoDet,
					      const MeasurementDet* stereoDet) :
  MeasurementDet(gdet), theGeomDet(gdet), 
  theMatcher(matcher), 
  theMonoDet( dynamic_cast<const TkStripMeasurementDet *>(monoDet)), 
  theStereoDet( dynamic_cast<const TkStripMeasurementDet *>(stereoDet))
{
  if ((theMonoDet == 0) || (theStereoDet == 0)) {
	throw MeasurementDetException("TkGluedMeasurementDet ERROR: Trying to glue a det which is not a TkStripMeasurementDet");
  }
  
}

TkGluedMeasurementDet::RecHitContainer 
TkGluedMeasurementDet::recHits( const TrajectoryStateOnSurface& ts) const
{

  RecHitContainer result;

  //------ WARNING: here ts is used as it is on the mono/stereo surface.
  //-----           A further propagation is necessary.
  //-----           To limit the problem, the SimpleCPE should be used
  RecHitContainer monoHits = theMonoDet->recHits( ts);
  RecHitContainer stereoHits = theStereoDet->recHits( ts);

  //edm::LogWarning("TkGluedMeasurementDet::recHits") << "Query-for-detid-" << theGeomDet->geographicalId().rawId();

  //checkProjection(ts, monoHits, stereoHits);

  if (monoHits.empty()) return projectOnGluedDet( stereoHits, ts);
  else if (stereoHits.empty()) return projectOnGluedDet(monoHits, ts);
  else {    
    LocalVector tkDir = (ts.isValid() ? ts.localDirection() : surface().toLocal( position()-GlobalPoint(0,0,0)));

    // convert stereo hits to type expected by matcher
    SiStripRecHitMatcher::SimpleHitCollection  vsStereoHits;
    vsStereoHits.reserve(stereoHits.size());
    for (RecHitContainer::const_iterator stereoHit = stereoHits.begin();
            stereoHit != stereoHits.end(); ++stereoHit) {
        const TrackingRecHit* tkhit = (**stereoHit).hit();
        const SiStripRecHit2D* verySpecificStereoHit =
            reinterpret_cast<const SiStripRecHit2D*>(tkhit);
        //dynamic_cast<const SiStripRecHit2D*>(tkhit);
        //if (verySpecificStereoHit == 0) {
        //   throw MeasurementDetException("TkGluedMeasurementDet ERROR: stereoHit is not SiStripRecHit2D");
        //}
        vsStereoHits.push_back( verySpecificStereoHit );
    }


    // convert mono hits to type expected by matcher
    for (RecHitContainer::const_iterator monoHit = monoHits.begin();
            monoHit != monoHits.end(); ++monoHit) {
        const TrackingRecHit* tkhit = (**monoHit).hit();
        const SiStripRecHit2D* verySpecificMonoHit =
            reinterpret_cast<const SiStripRecHit2D*>(tkhit);

        edm::OwnVector<SiStripMatchedRecHit2D> tmp = 
                theMatcher->match( verySpecificMonoHit, vsStereoHits.begin(), vsStereoHits.end(), 
                        &specificGeomDet(), tkDir);
  
        if (!tmp.empty()) {
            for (edm::OwnVector<SiStripMatchedRecHit2D>::const_iterator i=tmp.begin(), e = tmp.end();
                 i != e; ++i) {
              result.push_back( TSiStripMatchedRecHit::build( &geomDet(), &(*i), theMatcher));
            }
        } else {
            //<<<< if projecting 1 rec hit does not produce more than one rec hit ...
            //<<<< then the following is "suboptimal"
            //  RecHitContainer monoUnmatchedHit;     //better kept here, it is often never used at all in one call to recHits
            //  monoUnmatchedHit.push_back(*monoHit);
            //  RecHitContainer projectedMonoUnmatchedHit = projectOnGluedDet(monoUnmatchedHit, ts);
            //  result.insert(result.end(), projectedMonoUnmatchedHit.begin(), projectedMonoUnmatchedHit.end());
            //<<<< and so we use 
            TrackingRecHitProjector<ProjectedRecHit2D> proj;
            result.push_back( proj.project( **monoHit, geomDet(), ts));
            //<<<< end of change
        }
    } // loop on mono hit
  }
  return result;
}

std::vector<TrajectoryMeasurement> 
TkGluedMeasurementDet::fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
					 const TrajectoryStateOnSurface& startingState, 
					 const Propagator&, 
					 const MeasurementEstimator& est) const
{
   if (theMonoDet->isActive() || theStereoDet->isActive()) {
      NonPropagatingDetMeasurements realOne;
      
      if ( typeid(est) == typeid(Chi2MeasurementEstimator&) ) {
          Chi2MeasurementEstimatorForTrackerHits estOpt(static_cast<const Chi2MeasurementEstimator&>(est));;
          return realOne.get( *this, stateOnThisDet, estOpt);
          //return realOne.get( *this, stateOnThisDet, est);
      } else { // can't optimize for this
          return realOne.get( *this, stateOnThisDet, est);
      }
   } else {
      std::vector<TrajectoryMeasurement> result;
      result.push_back( TrajectoryMeasurement( stateOnThisDet, 
               InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 
               0.F));
      return result;	
   }

}

TkGluedMeasurementDet::RecHitContainer 
TkGluedMeasurementDet::projectOnGluedDet( const RecHitContainer& hits,
					  const TrajectoryStateOnSurface& ts) const
{
  if (hits.empty()) return hits;
  TrackingRecHitProjector<ProjectedRecHit2D> proj;
  RecHitContainer result;
  for ( RecHitContainer::const_iterator ihit = hits.begin(); ihit!=hits.end(); ihit++) {
    result.push_back( proj.project( **ihit, geomDet(), ts));
  }
  return result;
}


void TkGluedMeasurementDet::checkProjection(const TrajectoryStateOnSurface& ts, 
					    const RecHitContainer& monoHits, 
					    const RecHitContainer& stereoHits) const
{
  for (RecHitContainer::const_iterator i=monoHits.begin(); i != monoHits.end(); ++i) {
    checkHitProjection( **i, ts, geomDet());
  }
  for (RecHitContainer::const_iterator i=stereoHits.begin(); i != stereoHits.end(); ++i) {
    checkHitProjection( **i, ts, geomDet());
  }
}

void TkGluedMeasurementDet::checkHitProjection(const TransientTrackingRecHit& hit,
					       const TrajectoryStateOnSurface& ts, 
					       const GeomDet& det) const
{
  TrackingRecHitProjector<ProjectedRecHit2D> proj;
  TransientTrackingRecHit::RecHitPointer projectedHit = proj.project( hit, det, ts);

  RecHitPropagator prop;
  TrajectoryStateOnSurface propState = prop.propagate( hit, det.surface(), ts);

  if ((projectedHit->localPosition()-propState.localPosition()).mag() > 0.0001) {
    cout << "PROBLEM: projected and propagated hit positions differ by " 
	 << (projectedHit->localPosition()-propState.localPosition()).mag() << endl;
  }

  LocalError le1 = projectedHit->localPositionError();
  LocalError le2 = propState.localError().positionError();
  double eps = 1.e-5;
  double cutoff = 1.e-4; // if element below cutoff, use absolute instead of relative accuracy
  double maxdiff = std::max( std::max( fabs(le1.xx() - le2.xx())/(cutoff+le1.xx()),
				       fabs(le1.xy() - le2.xy())/(cutoff+fabs(le1.xy()))),
			     fabs(le1.yy() - le2.yy())/(cutoff+le1.xx()));  
  if (maxdiff > eps) { 
    cout << "PROBLEM: projected and propagated hit errors differ by " 
	 << maxdiff << endl;
  }
  
}

