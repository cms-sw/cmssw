#ifndef TkGluedMeasurementDet_H
#define TkGluedMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TkStripMeasurementDet.h"

class GluedGeomDet;
//class SiStripRecHitMatcher;
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include <TrackingTools/DetLayers/interface/MeasurementEstimator.h>
#include <TrackingTools/PatternTools/interface/TrajectoryMeasurement.h>


#include "FWCore/Utilities/interface/Visibility.h"


class TkGluedMeasurementDet GCC11_FINAL : public MeasurementDet {
public:

  TkGluedMeasurementDet( const GluedGeomDet* gdet,const SiStripRecHitMatcher* matcher, const StripClusterParameterEstimator* cpe);
  void init(const MeasurementDet* monoDet,
	    const MeasurementDet* stereoDet);

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&, const MeasurementTrackerEvent & data) const;

 // simple hits
  virtual bool recHits(SimpleHitContainer & result,  
		       const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator&, const MeasurementTrackerEvent & data) const;

  

 const GluedGeomDet& specificGeomDet() const {return static_cast<GluedGeomDet const&>(fastGeomDet());}

 virtual bool measurements( const TrajectoryStateOnSurface& stateOnThisDet,
			     const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
			    TempMeasurements & result) const;

  const TkStripMeasurementDet* monoDet() const{ return theMonoDet;} 
  const TkStripMeasurementDet* stereoDet() const{ return theStereoDet;} 

  /// return TRUE if both mono and stereo components are active
  bool isActive(const MeasurementTrackerEvent & data) const {return monoDet()->isActive(data) && stereoDet()->isActive(data); }
 	  	 
  /// return TRUE if at least one of the mono and stereo components has badChannels
  bool hasBadComponents( const TrajectoryStateOnSurface &tsos, const MeasurementTrackerEvent & data ) const {
    return (monoDet()->hasBadComponents(tsos,data) || stereoDet()->hasBadComponents(tsos,data));}

private:
  const SiStripRecHitMatcher*       theMatcher;
  const StripClusterParameterEstimator* theCPE;
  const TkStripMeasurementDet*       theMonoDet;
  const TkStripMeasurementDet*       theStereoDet;


  template<typename Collector>
  void doubleMatch(const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data, Collector & collector) const  dso_internal;

  template<typename Collector>
  void collectRecHits(const TrajectoryStateOnSurface&, const MeasurementTrackerEvent & data, Collector &coll) const dso_internal;

  // for TTRH
  class dso_internal  HitCollectorForRecHits {
  public:
    typedef SiStripRecHitMatcher::Collector Collector;
    HitCollectorForRecHits(const GeomDet * geomDet, 
			   const SiStripRecHitMatcher * matcher,
			   const StripClusterParameterEstimator* cpe,
			   RecHitContainer & target) ;
    void add(SiStripMatchedRecHit2D const& hit) {
      target_.emplace_back(hit.cloneSH());
      hasNewHits_ = true; 
    }
    void addProjected(const TrackingRecHit& hit,
		      const GlobalVector & gdir) ;
    SiStripRecHitMatcher::Collector & collector() { return collector_; }
    bool hasNewMatchedHits() const { return hasNewHits_;  }
    void clearNewMatchedHitsFlag() { hasNewHits_ = false; }
    static bool filter() { return false;}   /// always fast as no estimator available here! 
    size_t size() const { return target_.size();}

    static const MeasurementEstimator  & estimator() { static MeasurementEstimator * dummy=0; return *dummy;}

  private: 
    const GeomDet              * geomDet_;
    const SiStripRecHitMatcher * matcher_;
    const StripClusterParameterEstimator* cpe_;
    RecHitContainer       & target_;
    SiStripRecHitMatcher::Collector collector_;       
    bool hasNewHits_;
  };

  // for TRH
  class dso_internal  HitCollectorForSimpleHits {
  public:
    typedef SiStripRecHitMatcher::Collector Collector;
    HitCollectorForSimpleHits(const GeomDet * geomDet, 
			      const SiStripRecHitMatcher * matcher,
			      const StripClusterParameterEstimator* cpe,
			      const TrajectoryStateOnSurface& stateOnThisDet,
			      const MeasurementEstimator& est,
			      SimpleHitContainer & target) ;
    void add(SiStripMatchedRecHit2D const & hit);
    void addProjected(const TrackingRecHit& hit,
		      const GlobalVector & gdir) ;
    SiStripRecHitMatcher::Collector & collector() { return collector_; }
    bool hasNewMatchedHits() const { return hasNewHits_;  }
    void clearNewMatchedHitsFlag() { hasNewHits_ = false; }
    bool filter() const { return matcher_->preFilter();}   // if true mono-colection will been filter using the estimator before matching  
    size_t size() const { return target_.size();}
    const MeasurementEstimator  & estimator() { return est_;}
  private: 
    const GeomDet              * geomDet_;
    const SiStripRecHitMatcher * matcher_;
    const StripClusterParameterEstimator* cpe_;
    const TrajectoryStateOnSurface & stateOnThisDet_;
    const MeasurementEstimator     & est_;
    SimpleHitContainer & target_;
    SiStripRecHitMatcher::Collector collector_;       
    bool hasNewHits_;
  };

  

  class dso_internal HitCollectorForFastMeasurements {
  public:
    typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
    typedef SiStripRecHitMatcher::Collector Collector;
    
    HitCollectorForFastMeasurements(const GeomDet * geomDet, 
				    const SiStripRecHitMatcher * matcher,
				    const StripClusterParameterEstimator* cpe,
				    const TrajectoryStateOnSurface& stateOnThisDet,
				    const MeasurementEstimator& est,
				    TempMeasurements & target) ;
    void add(SiStripMatchedRecHit2D const& hit) ;
    void addProjected(const TrackingRecHit& hit,
		      const GlobalVector & gdir) ;
    
    SiStripRecHitMatcher::Collector & collector() { return collector_; }
    bool hasNewMatchedHits() const { return hasNewHits_;  }
    void clearNewMatchedHitsFlag() { hasNewHits_ = false; }
    bool filter() const { return matcher_->preFilter();}   // if true mono-colection will been filter using the estimator before matching  
    size_t size() const { return target_.size();}
    const MeasurementEstimator  & estimator() { return est_;}
  private: 
    const GeomDet              * geomDet_;
    const SiStripRecHitMatcher * matcher_;
    const StripClusterParameterEstimator* cpe_;
    const TrajectoryStateOnSurface & stateOnThisDet_;
    const MeasurementEstimator     & est_;
    TempMeasurements & target_;
    SiStripRecHitMatcher::Collector collector_;       
    bool hasNewHits_;
  };
  

  
  RecHitContainer 
  projectOnGluedDet( const std::vector<SiStripRecHit2D>& hits,
		     const TrajectoryStateOnSurface& ts) const dso_internal;
  template<typename HitCollector>
  void
  projectOnGluedDet( HitCollector & collector,
                     const std::vector<SiStripRecHit2D>& hits,
                     const GlobalVector & gdir ) const  dso_internal;


  RecHitContainer 
    projectOnGluedDet( const RecHitContainer& hits,
		       const TrajectoryStateOnSurface& ts) const dso_internal;
  template<typename HitCollector>
  void
  projectOnGluedDet( HitCollector & collector,
                     const RecHitContainer& hits,
                     const GlobalVector & gdir ) const  dso_internal;


  void checkProjection(const TrajectoryStateOnSurface& ts, 
		       const RecHitContainer& monoHits, 
		       const RecHitContainer& stereoHits) const;
  void checkHitProjection(const TrackingRecHit& hit,
			  const TrajectoryStateOnSurface& ts, 
			  const GeomDet& det) const dso_internal;

  /** \brief Test the strips on one of the two dets with projection */
  bool testStrips(const TrajectoryStateOnSurface& tsos,
                  const BoundPlane &gluedPlane,
                  const TkStripMeasurementDet &mdet) const  dso_internal;

};

#endif
