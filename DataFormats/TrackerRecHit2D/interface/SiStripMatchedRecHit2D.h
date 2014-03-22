#ifndef SiStripMatchedRecHit2D_H
#define SiStripMatchedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "TkCloner.h"

class SiStripMatchedRecHit2D GCC11_FINAL : public BaseTrackerRecHit {
 public:
  typedef BaseTrackerRecHit Base;

  SiStripMatchedRecHit2D(){}
  ~SiStripMatchedRecHit2D(){}

  SiStripMatchedRecHit2D( const LocalPoint& pos, const LocalError& err, GeomDet const & idet,
			  const SiStripRecHit2D* rMono,const SiStripRecHit2D* rStereo):
    BaseTrackerRecHit(pos, err, idet, trackerHitRTTI::match), clusterMono_(rMono->omniClusterRef()), clusterStereo_(rStereo->omniClusterRef()){}

  // by value, as they will not exists anymore...
  SiStripRecHit2D  stereoHit() const { return SiStripRecHit2D(stereoId(),stereoClusterRef()) ;}
  SiStripRecHit2D  monoHit() const { return SiStripRecHit2D(monoId(),monoClusterRef());}
 
  unsigned int stereoId() const { return rawId()+1;}
  unsigned int monoId()   const { return rawId()+2;}

  // (to be improved)
  virtual OmniClusterRef const & firstClusterRef() const { return monoClusterRef();}


  OmniClusterRef const & stereoClusterRef() const { return clusterStereo_;}
  OmniClusterRef const & monoClusterRef() const { return clusterMono_;}
  // Non const variants needed for cluster re-keying 
  OmniClusterRef & stereoClusterRef()  { return clusterStereo_;}
  OmniClusterRef  & monoClusterRef()  { return clusterMono_;}
  
  SiStripCluster const & stereoCluster() const { 
    return stereoClusterRef().stripCluster();
  }  
  SiStripCluster const & monoCluster() const { 
    return monoClusterRef().stripCluster();
  }  


  virtual SiStripMatchedRecHit2D * clone() const {return new SiStripMatchedRecHit2D( * this);}
 
  virtual int dimension() const {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const { getKfComponents2D(holder); }



  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;

  bool sharesInput(TrackerSingleRecHit const & other) const;

  virtual std::vector<const TrackingRecHit*> recHits() const; 

  virtual std::vector<TrackingRecHit*> recHits(); 

  virtual bool canImproveWithTrack() const {return true;}
private:
  // double dispatch
  virtual SiStripMatchedRecHit2D * clone(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const {
    return cloner(*this,tsos);
  }
 
    
 private:
   OmniClusterRef clusterMono_, clusterStereo_;
};


inline 
bool sharesClusters(SiStripMatchedRecHit2D const & h1, SiStripMatchedRecHit2D const & h2,
		    TrackingRecHit::SharedInputType what) {
  bool mono =  h1.monoClusterRef()== h2.monoClusterRef();
  bool stereo =  h1.stereoClusterRef()== h2.stereoClusterRef();
  
  return (what==TrackingRecHit::all) ? (mono&&stereo) : (mono||stereo);
  
} 

#endif
