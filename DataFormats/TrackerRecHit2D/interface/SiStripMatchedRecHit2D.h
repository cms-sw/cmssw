#ifndef SiStripMatchedRecHit2D_H
#define SiStripMatchedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

class SiStripMatchedRecHit2D : public BaseTrackerRecHit {
 public:
  typedef BaseTrackerRecHit Base;

  SiStripMatchedRecHit2D(){}
  ~SiStripMatchedRecHit2D(){}
  SiStripMatchedRecHit2D( const LocalPoint& pos, const LocalError& err, const DetId& id , 
			  const SiStripRecHit2D* rMono,const SiStripRecHit2D* rStereo):
    BaseTrackerRecHit(pos, err, id, trackerHitRTTI::match), componentMono_(*rMono),componentStereo_(*rStereo){}
					 
  const SiStripRecHit2D *stereoHit() const { return &componentStereo_;}
  const SiStripRecHit2D *monoHit() const { return &componentMono_;}
 
  // Non const variants needed for cluster re-keying 
  // SiStripRecHit2D *stereoHit() { return &componentStereo_;}
  // SiStripRecHit2D *monoHit() { return &componentMono_;}

   // used by trackMerger (to be improved)
  virtual OmniClusterRef const & firstClusterRef() const { return monoClusterRef();}


  OmniClusterRef const & stereoClusterRef() const { return componentStereo_.omniCluster();}
  OmniClusterRef const & monoClusterRef() const { return componentMono_.omniCluster();}
  // Non const variants needed for cluster re-keying 
  OmniClusterRef & stereoClusterRef()  { return componentStereo_.omniCluster();}
  OmniClusterRef  & monoClusterRef()  { return componentMono_.omniCluster();}
  
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

    
 private:
  SiStripRecHit2D componentMono_, componentStereo_;
};


#endif
