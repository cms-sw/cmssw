#ifndef TkGluedMeasurementDet_H
#define TkGluedMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"

class GluedGeomDet;
//class SiStripRecHitMatcher;
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include <TrackingTools/PatternTools/interface/MeasurementEstimator.h>
#include <TrackingTools/PatternTools/interface/TrajectoryMeasurement.h>

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

  const TkStripMeasurementDet* monoDet() const{ return theMonoDet;} 
  const TkStripMeasurementDet* stereoDet() const{ return theStereoDet;} 

private:
  const GluedGeomDet*         theGeomDet;
  const SiStripRecHitMatcher*       theMatcher;
  const TkStripMeasurementDet*       theMonoDet;
  const TkStripMeasurementDet*       theStereoDet;


  template<typename Collector>
  void doubleMatch(const TrajectoryStateOnSurface& ts, Collector & collector) const;

  template<typename Collector>
  void collectRecHits(const TrajectoryStateOnSurface&, Collector &coll) const;

  class HitCollectorForRecHits {
    public:
       HitCollectorForRecHits(const GeomDet * geomDet, 
                              const SiStripRecHitMatcher * matcher,
                              RecHitContainer & target) ;
       void add(SiStripMatchedRecHit2D const& hit) {
           target_.push_back(
                TSiStripMatchedRecHit::build( geomDet_, std::auto_ptr<TrackingRecHit>(hit.clone()), matcher_)
           );
           hasNewHits_ = true; 
       }
       void addProjected(const TransientTrackingRecHit& hit,
                         const GlobalVector & gdir) ;
       SiStripRecHitMatcher::Collector & collector() { return collector_; }
       bool hasNewMatchedHits() const { return hasNewHits_;  }
       void clearNewMatchedHitsFlag() { hasNewHits_ = false; }
    private: 
       const GeomDet              * geomDet_;
       const SiStripRecHitMatcher * matcher_;
       RecHitContainer       & target_;
       SiStripRecHitMatcher::Collector collector_;       
       bool hasNewHits_;
  };


  class HitCollectorForFastMeasurements {
    public:
       typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
       HitCollectorForFastMeasurements(const GeomDet * geomDet, 
                              const SiStripRecHitMatcher * matcher,
                              const TrajectoryStateOnSurface& stateOnThisDet,
                              const MeasurementEstimator& est,
                              std::vector<TrajectoryMeasurement> & target) ;
       void add(SiStripMatchedRecHit2D const& hit) ;
       void addProjected(const TransientTrackingRecHit& hit,
                         const GlobalVector & gdir) ;

       SiStripRecHitMatcher::Collector & collector() { return collector_; }
       bool hasNewMatchedHits() const { return hasNewHits_;  }
       void clearNewMatchedHitsFlag() { hasNewHits_ = false; }
    private: 
       const GeomDet              * geomDet_;
       const SiStripRecHitMatcher * matcher_;
       const TrajectoryStateOnSurface & stateOnThisDet_;
       const MeasurementEstimator     & est_;
       std::vector<TrajectoryMeasurement> & target_;
       SiStripRecHitMatcher::Collector collector_;       
       bool hasNewHits_;
  };


  RecHitContainer 
  projectOnGluedDet( const RecHitContainer& hits,
		     const TrajectoryStateOnSurface& ts) const;

  template<typename HitCollector>
  void
  projectOnGluedDet( HitCollector & collector, 
                     const RecHitContainer& hits,
                     const GlobalVector & gdir ) const ;

  void checkProjection(const TrajectoryStateOnSurface& ts, 
		       const RecHitContainer& monoHits, 
		       const RecHitContainer& stereoHits) const;
  void checkHitProjection(const TransientTrackingRecHit& hit,
			  const TrajectoryStateOnSurface& ts, 
			  const GeomDet& det) const;

  /** \brief Test the strips on one of the two dets with projection */
  bool testStrips(const TrajectoryStateOnSurface& tsos,
                  const BoundPlane &gluedPlane,
                  const TkStripMeasurementDet &mdet) const ;

};

#endif
