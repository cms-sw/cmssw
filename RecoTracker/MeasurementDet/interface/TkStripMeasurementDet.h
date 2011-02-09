#ifndef TkStripMeasurementDet_H
#define TkStripMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TransientTrackingRecHit;

class TkStripMeasurementDet : public MeasurementDet {
public:

  //  typedef SiStripClusterCollection::Range                ClusterRange;
  //  typedef SiStripClusterCollection::ContainerIterator    ClusterIterator;
  typedef StripClusterParameterEstimator::LocalValues    LocalValues;
  typedef StripClusterParameterEstimator::VLocalValues    VLocalValues;

  typedef SiStripRecHit2D::ClusterRef SiStripClusterRef;

  typedef edm::LazyGetter<SiStripCluster>::value_ref  SiStripRegionalClusterRef;

  typedef edmNew::DetSet<SiStripCluster> detset;
  typedef detset::const_iterator new_const_iterator;

  typedef std::vector<SiStripCluster>::const_iterator const_iterator;

  virtual ~TkStripMeasurementDet(){}

  TkStripMeasurementDet( const GeomDet* gdet,
			 const StripClusterParameterEstimator* cpe,
			 bool regional);

  void update( const detset &detSet, 
	       const edm::Handle<edmNew::DetSetVector<SiStripCluster> > h,
	       unsigned int id ) { 
    detSet_ = detSet; 
    handle_ = h;
    id_ = id;
    empty = false;
    isRegional = false;
  }

  void update( std::vector<SiStripCluster>::const_iterator begin ,std::vector<SiStripCluster>::const_iterator end, 
	       const edm::Handle<edm::LazyGetter<SiStripCluster> > h,
	       unsigned int id ) { 
    beginCluster = begin;
    endCluster   = end;
    regionalHandle_ = h;
    id_ = id;
    empty = false;
    activeThisEvent_ = true;
    isRegional = true;
  }
  
  /** \brief Is this module active in reconstruction? It must be both 'setActiveThisEvent' and 'setActive'. */
  bool isActive() const { return activeThisEvent_ && activeThisPeriod_; }
 	  	 
  //TO BE IMPLEMENTED
  bool hasBadComponents( const TrajectoryStateOnSurface &tsos ) const {return false;}


  void setEmpty(){empty = true; activeThisEvent_ = true; }
  
  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&) const;
  void simpleRecHits( const TrajectoryStateOnSurface& ts, std::vector<SiStripRecHit2D> &result) const ;

  virtual std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& startingState, 
		    const Propagator&, 
		    const MeasurementEstimator&) const;

  const StripGeomDetUnit& specificGeomDet() const {return *theStripGDU;}

  TransientTrackingRecHit::RecHitPointer
  buildRecHit( const SiStripClusterRef&, const TrajectoryStateOnSurface& ltp) const;

  TransientTrackingRecHit::RecHitPointer
  buildRecHit( const SiStripRegionalClusterRef&, const TrajectoryStateOnSurface& ltp) const;

    
  TkStripMeasurementDet::RecHitContainer 
  buildRecHits( const SiStripClusterRef&, const TrajectoryStateOnSurface& ltp) const;
  
  TkStripMeasurementDet::RecHitContainer 
  buildRecHits( const SiStripRegionalClusterRef&, const TrajectoryStateOnSurface& ltp) const;


  bool  isEmpty() {return empty;}

  unsigned int rawId() const { return id_; }


  const detset& theSet() {return detSet_;}
  int  size() {return endCluster - beginCluster ; }

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually).
             This also resets the 'setActiveThisEvent' to true */
  void setActive(bool active) { activeThisPeriod_ = active; activeThisEvent_ = true; if (!active) empty = true; }
  /** \brief Turn on/off the module for reconstruction for one events.
             This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(bool active) { activeThisEvent_ = active;  if (!active) empty = true; }

  /** \brief does this module have at least one bad strip, APV or channel? */
  bool hasAllGoodChannels() const { return !hasAny128StripBad_ && badStripBlocks_.empty(); }

  /** \brief Sets the status of a block of 128 strips (or all blocks if idx=-1) */
  void set128StripStatus(bool good, int idx=-1);

  struct BadStripCuts {
     BadStripCuts() : maxBad(9999), maxConsecutiveBad(9999) {}
     BadStripCuts(const edm::ParameterSet &pset) :
        maxBad(pset.getParameter<uint32_t>("maxBad")),
        maxConsecutiveBad(pset.getParameter<uint32_t>("maxConsecutiveBad")) {}
     uint16_t maxBad, maxConsecutiveBad;
  };

  /** \brief return true if there are 'enough' good strips in the utraj +/- 3 uerr range.*/
  bool testStrips(float utraj, float uerr) const;

  void setBadStripCuts(BadStripCuts cuts) { badStripCuts_ = cuts; }

  struct BadStripBlock {
      short first;
      short last;
      BadStripBlock(const SiStripBadStrip::data &data) : first(data.firstStrip), last(data.firstStrip+data.range-1) { }
  };
  std::vector<BadStripBlock> &getBadStripBlocks() { return badStripBlocks_; }

  void setMaskBad128StripBlocks(bool maskThem) { maskBad128StripBlocks_ = maskThem; }

private:

  bool isRegional;

  bool empty;

  bool activeThisEvent_,activeThisPeriod_;

  unsigned int id_;

  detset detSet_;
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > handle_;


  const StripGeomDetUnit*               theStripGDU;
  const StripClusterParameterEstimator* theCPE;



  bool bad128Strip_[6];
  bool hasAny128StripBad_, maskBad128StripBlocks_;

  std::vector<BadStripBlock> badStripBlocks_;  
  int totalStrips_;
  BadStripCuts badStripCuts_;
 

  // --- regional unpacking
  edm::Handle<edm::LazyGetter<SiStripCluster> > regionalHandle_;
  std::vector<SiStripCluster>::const_iterator beginCluster;
  std::vector<SiStripCluster>::const_iterator endCluster;
  // regional unpacking ---

  inline bool isMasked(const SiStripCluster &cluster) const {
      if ( bad128Strip_[cluster.firstStrip() >> 7] ) {
          if ( bad128Strip_[(cluster.firstStrip()+cluster.amplitudes().size())  >> 7] ||
               bad128Strip_[static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7] ) {
              return true;
          }
      } else {
          if ( bad128Strip_[(cluster.firstStrip()+cluster.amplitudes().size())  >> 7] &&
               bad128Strip_[static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7] ) {
              return true;
          }
      }
      return false;
  }
  
  template<class ClusterRefT>
    void buildSimpleRecHit( const ClusterRefT& cluster,
			    const TrajectoryStateOnSurface& ltp,
			    std::vector<SiStripRecHit2D>& res) const;
  
};

#endif
