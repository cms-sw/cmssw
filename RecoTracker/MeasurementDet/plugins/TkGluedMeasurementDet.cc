#include "TkGluedMeasurementDet.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "NonPropagatingDetMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "RecHitPropagator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

#include <typeinfo>

// #include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

using namespace std;

TkGluedMeasurementDet::TkGluedMeasurementDet( const GluedGeomDet* gdet, 
					      const SiStripRecHitMatcher* matcher) :
  MeasurementDet(gdet), 
  theMatcher(matcher),
  theMonoDet(nullptr), theStereoDet(nullptr)
{}

void TkGluedMeasurementDet::init(const MeasurementDet* monoDet,
				 const MeasurementDet* stereoDet) {
  theMonoDet = dynamic_cast<const TkStripMeasurementDet *>(monoDet);
  theStereoDet = dynamic_cast<const TkStripMeasurementDet *>(stereoDet);
  
  if ((theMonoDet == 0) || (theStereoDet == 0)) {
    throw MeasurementDetException("TkGluedMeasurementDet ERROR: Trying to glue a det which is not a TkStripMeasurementDet");
  }
}

TkGluedMeasurementDet::RecHitContainer 
TkGluedMeasurementDet::recHits( const TrajectoryStateOnSurface& ts) const
{

  RecHitContainer result;
  HitCollectorForRecHits collector( &fastGeomDet(), theMatcher, result );
  collectRecHits(ts, collector);
  return result;
}

struct take_address { template<typename T> const T * operator()(const T &val) const { return &val; } };

#ifdef DOUBLE_MATCH
template<typename Collector>
void
TkGluedMeasurementDet::collectRecHits( const TrajectoryStateOnSurface& ts, Collector & collector) const
{
  doubleMatch(ts,collector);
}
#else
template<typename Collector>
void
TkGluedMeasurementDet::collectRecHits( const TrajectoryStateOnSurface& ts, Collector & collector) const
{
  //------ WARNING: here ts is used as it is on the mono/stereo surface.
  //-----           A further propagation is necessary.
  //-----           To limit the problem, the SimpleCPE should be used
  RecHitContainer monoHits = theMonoDet->recHits( ts);
  GlobalVector glbDir = (ts.isValid() ? ts.globalParameters().momentum() : position()-GlobalPoint(0,0,0));

  //edm::LogWarning("TkGluedMeasurementDet::recHits") << "Query-for-detid-" << theGeomDet->geographicalId().rawId();

  //checkProjection(ts, monoHits, stereoHits);

  if (monoHits.empty()) {
      // make stereo TTRHs and project them
      projectOnGluedDet( collector, theStereoDet->recHits(ts), glbDir);
  } else {
      // collect simple stereo hits
      static std::vector<SiStripRecHit2D> simpleSteroHitsByValue;
      simpleSteroHitsByValue.clear();
      theStereoDet->simpleRecHits(ts, simpleSteroHitsByValue);

      if (simpleSteroHitsByValue.empty()) {
          projectOnGluedDet( collector, monoHits, glbDir);
      } else {

          LocalVector tkDir = (ts.isValid() ? ts.localDirection() : surface().toLocal( position()-GlobalPoint(0,0,0)));
          static SiStripRecHitMatcher::SimpleHitCollection vsStereoHits;
          vsStereoHits.resize(simpleSteroHitsByValue.size());
          std::transform(simpleSteroHitsByValue.begin(), simpleSteroHitsByValue.end(), vsStereoHits.begin(), take_address()); 

          // convert mono hits to type expected by matcher
          for (RecHitContainer::const_iterator monoHit = monoHits.begin();
                  monoHit != monoHits.end(); ++monoHit) {
              const TrackingRecHit* tkhit = (**monoHit).hit();
              const SiStripRecHit2D* verySpecificMonoHit = reinterpret_cast<const SiStripRecHit2D*>(tkhit);
              theMatcher->match( verySpecificMonoHit, vsStereoHits.begin(), vsStereoHits.end(), 
                      collector.collector(), &specificGeomDet(), tkDir);

              if (collector.hasNewMatchedHits()) {
                  collector.clearNewMatchedHitsFlag();
              } else {
                  collector.addProjected( **monoHit, glbDir );
              }
          } // loop on mono hit
      }
      //GIO// std::cerr << "TkGluedMeasurementDet hits " << monoHits.size() << "/" << stereoHits.size() << " => " << result.size() << std::endl;
  }
}
#endif

std::vector<TrajectoryMeasurement> 
TkGluedMeasurementDet::fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
					 const TrajectoryStateOnSurface& startingState, 
					 const Propagator&, 
					 const MeasurementEstimator& est) const
{
   std::vector<TrajectoryMeasurement> result;
   if (theMonoDet->isActive() || theStereoDet->isActive()) {

      HitCollectorForFastMeasurements collector( &fastGeomDet(), theMatcher, stateOnThisDet, est, result);
      collectRecHits(stateOnThisDet, collector);
       
      if ( result.empty()) {
          //LogDebug("TkStripMeasurementDet") << "No hit found on TkGlued. Testing strips...  ";
          const BoundPlane &gluedPlane = geomDet().surface();
          if (  // sorry for the big IF, but I want to exploit short-circuiting of logic
               stateOnThisDet.hasError() && ( /* do this only if the state has uncertainties, otherwise it will throw 
                                                 (states without uncertainties are passed to this code from seeding */
                (theMonoDet->isActive() && 
                    (theMonoDet->hasAllGoodChannels() || 
                       testStrips(stateOnThisDet,gluedPlane,*theMonoDet)
                    )
                ) /*Mono OK*/ || 
                (theStereoDet->isActive() && 
                    (theStereoDet->hasAllGoodChannels() || 
                       testStrips(stateOnThisDet,gluedPlane,*theStereoDet)
                    )
                ) /*Stereo OK*/ 
               ) /* State has errors */
              ) {
              result.push_back( TrajectoryMeasurement( stateOnThisDet, 
                          InvalidTransientRecHit::build(&fastGeomDet()), 0.F)); 
          } else {
              result.push_back( TrajectoryMeasurement(stateOnThisDet, 
                         InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::inactive), 0.F));
          }
      } else {
          // sort results according to estimator value
          if ( result.size() > 1) {
              sort( result.begin(), result.end(), TrajMeasLessEstim());
          }
      }
   } else {
     //     LogDebug("TkStripMeasurementDet") << " DetID " << geomDet().geographicalId().rawId() << " (glued) fully inactive";
      result.push_back( TrajectoryMeasurement( stateOnThisDet, 
               InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::inactive), 
               0.F));
   }
   return result;	

}

TkGluedMeasurementDet::RecHitContainer 
TkGluedMeasurementDet::projectOnGluedDet( const RecHitContainer& hits,
					  const TrajectoryStateOnSurface& ts) const
{
  if (hits.empty()) return hits;
  TrackingRecHitProjector<ProjectedRecHit2D> proj;
  RecHitContainer result;
  for ( RecHitContainer::const_iterator ihit = hits.begin(); ihit!=hits.end(); ihit++) {
    result.push_back( proj.project( **ihit, fastGeomDet(), ts));
  }
  return result;
}

template<typename Collector>
void 
TkGluedMeasurementDet::projectOnGluedDet( Collector& collector,
                                          const RecHitContainer& hits,
                                          const GlobalVector & gdir ) const 
{
  for ( RecHitContainer::const_iterator ihit = hits.begin(); ihit!=hits.end(); ihit++) {
    collector.addProjected( **ihit, gdir );
  }
}

void TkGluedMeasurementDet::checkProjection(const TrajectoryStateOnSurface& ts, 
					    const RecHitContainer& monoHits, 
					    const RecHitContainer& stereoHits) const
{
  for (RecHitContainer::const_iterator i=monoHits.begin(); i != monoHits.end(); ++i) {
    checkHitProjection( **i, ts, fastGeomDet());
  }
  for (RecHitContainer::const_iterator i=stereoHits.begin(); i != stereoHits.end(); ++i) {
    checkHitProjection( **i, ts, fastGeomDet());
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

bool
TkGluedMeasurementDet::testStrips(const TrajectoryStateOnSurface& tsos,
                                  const BoundPlane &gluedPlane,
                                  const TkStripMeasurementDet &mdet) const {
   // from TrackingRecHitProjector
   const GeomDet &det = mdet.fastGeomDet();
   const BoundPlane &stripPlane = det.surface();

   //LocalPoint glp = tsos.localPosition();
   LocalError  err = tsos.localError().positionError();
   /*LogDebug("TkStripMeasurementDet") << 
      "Testing local pos glued: " << glp << 
      " local err glued: " << tsos.localError().positionError() << 
      " in? " << gluedPlane.bounds().inside(glp) <<
      " in(3s)? " << gluedPlane.bounds().inside(glp, err, 3.0f);*/

   GlobalVector gdir = tsos.globalParameters().momentum();

   LocalPoint  slp = stripPlane.toLocal(tsos.globalPosition()); 
   LocalVector sld = stripPlane.toLocal(gdir);

   double delta = stripPlane.localZ( tsos.globalPosition());
   LocalPoint pos = slp - sld * delta/sld.z();


   // now the error
   LocalVector hitXAxis = stripPlane.toLocal( gluedPlane.toGlobal( LocalVector(1,0,0)));
   if (stripPlane.normalVector().dot( gluedPlane.normalVector()) < 0) {
       // the two planes are inverted, and the correlation element must change sign
       err = LocalError( err.xx(), -err.xy(), err.yy());
   }
   LocalError rotatedError = err.rotate( hitXAxis.x(), hitXAxis.y());

   /* // This is probably meaningless 
   LogDebug("TkStripMeasurementDet") << 
      "Testing local pos on strip (SLP): " << slp << 
      " in? :" << stripPlane.bounds().inside(slp) <<
      " in(3s)? :" << stripPlane.bounds().inside(slp, rotatedError, 3.0f);
   // but it helps to test bugs in the formula for POS */
   /*LogDebug("TkStripMeasurementDet") << 
      "Testing local pos strip: " << pos << 
      " in? " << stripPlane.bounds().inside(pos) <<
      " in(3s)? " << stripPlane.bounds().inside(pos, rotatedError, 3.0f);*/

   // now we need to convert to MeasurementFrame
   const StripTopology &topo = mdet.specificGeomDet().specificTopology();
   float utraj = topo.measurementPosition(pos).x();
   float uerr  = std::sqrt(topo.measurementError(pos,rotatedError).uu());
   return mdet.testStrips(utraj, uerr);
} 

#include<boost/bind.hpp>
TkGluedMeasurementDet::HitCollectorForRecHits::HitCollectorForRecHits(const GeomDet * geomDet, 
        const SiStripRecHitMatcher * matcher,
        RecHitContainer & target) :
    geomDet_(geomDet), matcher_(matcher), target_(target),
    collector_(boost::bind(&HitCollectorForRecHits::add,boost::ref(*this),_1)),
    hasNewHits_(false)
{
}

void
TkGluedMeasurementDet::HitCollectorForRecHits::addProjected(const TransientTrackingRecHit& hit,
                                                            const GlobalVector & gdir)
{
    TrackingRecHitProjector<ProjectedRecHit2D> proj;
    target_.push_back( proj.project( hit, *geomDet_, gdir));
}

TkGluedMeasurementDet::HitCollectorForFastMeasurements::HitCollectorForFastMeasurements(const GeomDet * geomDet, 
        const SiStripRecHitMatcher * matcher,
        const TrajectoryStateOnSurface& stateOnThisDet,
        const MeasurementEstimator& est,
        std::vector<TrajectoryMeasurement> & target) :
    geomDet_(geomDet), matcher_(matcher), stateOnThisDet_(stateOnThisDet), est_(est), target_(target),
    collector_(boost::bind(&HitCollectorForFastMeasurements::add,boost::ref(*this),_1)),
    hasNewHits_(false)
{
}

void
TkGluedMeasurementDet::HitCollectorForFastMeasurements::add(SiStripMatchedRecHit2D const& hit2d) 
{
  static LocalCache<TSiStripMatchedRecHit> lcache; // in case of pool allocator it will be cleared centrally
  std::auto_ptr<TSiStripMatchedRecHit> & cache = lcache.ptr;
  TSiStripMatchedRecHit::buildInPlace( cache, geomDet_, &hit2d, matcher_ );
  std::pair<bool,double> diffEst = est_.estimate( stateOnThisDet_, *cache);
  if ( diffEst.first) {
    cache->clonePersistentHit(); // clone and take ownership of the persistent 2D hit
    target_.push_back( 
		      TrajectoryMeasurement( stateOnThisDet_, 
					     RecHitPointer(cache.release()), 
					     diffEst.second)
		       );
  } else {
    cache->clearPersistentHit(); // drop ownership
  } 
  hasNewHits_ = true; //FIXME: see also what happens moving this within testAndPush
}

void
TkGluedMeasurementDet::HitCollectorForFastMeasurements::addProjected(const TransientTrackingRecHit& hit,
                                                            const GlobalVector & gdir)
{
    // here we're ok with some extra casual new's and delete's
    TrackingRecHitProjector<ProjectedRecHit2D> proj;
    RecHitPointer phit = proj.project( hit, *geomDet_, gdir );
    std::pair<bool,double> diffEst = est_.estimate( stateOnThisDet_, *phit);
    if ( diffEst.first) {
        target_.push_back( TrajectoryMeasurement( stateOnThisDet_, phit, diffEst.second) );
    }

}



#ifdef DOUBLE_MATCH
#include "doubleMatch.icc"
#endif
