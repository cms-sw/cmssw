#ifndef SiStripMatchedRecHit2D_H
#define SiStripMatchedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "TkCloner.h"

class SiStripMatchedRecHit2D final : public BaseTrackerRecHit {
 public:
  typedef BaseTrackerRecHit Base;

  SiStripMatchedRecHit2D(){}
  ~SiStripMatchedRecHit2D() override{}

  SiStripMatchedRecHit2D( const LocalPoint& pos, const LocalError& err, GeomDet const & idet,
			  const SiStripRecHit2D* rMono,const SiStripRecHit2D* rStereo):
    BaseTrackerRecHit(pos, err, idet, trackerHitRTTI::match), clusterMono_(rMono->omniClusterRef()), clusterStereo_(rStereo->omniClusterRef()){}

  // by value, as they will not exists anymore...
  SiStripRecHit2D  stereoHit() const { return SiStripRecHit2D(stereoId(),stereoClusterRef()) ;}
  SiStripRecHit2D  monoHit() const { return SiStripRecHit2D(monoId(),monoClusterRef());}
 
  unsigned int stereoId() const { return rawId()+1;}
  unsigned int monoId()   const { return rawId()+2;}

  // (to be improved)
  OmniClusterRef const & firstClusterRef() const override { return monoClusterRef();}


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


  SiStripMatchedRecHit2D * clone() const override {return new SiStripMatchedRecHit2D( * this);}
#ifndef __GCCXML__
  RecHitPointer cloneSH() const override { return std::make_shared<SiStripMatchedRecHit2D>(*this);}
#endif

 
  int dimension() const override {return 2;}
  void getKfComponents( KfComponentsHolder & holder ) const override { getKfComponents2D(holder); }



  bool sharesInput( const TrackingRecHit* other, SharedInputType what) const override;

  bool sharesInput(TrackerSingleRecHit const & other) const;

  std::vector<const TrackingRecHit*> recHits() const override; 

  std::vector<TrackingRecHit*> recHits() override; 

  bool canImproveWithTrack() const override {return true;}
private:
  // double dispatch
  SiStripMatchedRecHit2D * clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner(*this,tsos).release();
  }
#ifndef __GCCXML__
   RecHitPointer cloneSH_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner.makeShared(*this,tsos);
  }
#endif 
    
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
