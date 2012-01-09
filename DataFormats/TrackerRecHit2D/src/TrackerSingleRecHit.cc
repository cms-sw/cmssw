#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"



#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include<iostream>
namespace {
  
  void verify(OmniClusterRef const ref) {
    std::cout << 
      ref.rawIndex() << " " <<
      ref.isValid() << " " <<
      ref.isPixel() << " " <<
      ref.isStrip()  << " " <<
      ref.isRegional() << " " <<
      ref.cluster_strip().isNull() << " " <<
      ref.cluster_pixel().isNull() << " " <<
      ref.cluster_regional().isNull()  << " " << std::endl;
  }
  
  void verify(TrackingRecHit const * thit) {

    static bool once=false;
    if (once) {
      once=true;
      DetID Lim(DetId::Tracker,3);
      std::cout << "detid lim " << (Lim..rawId() >> (DetId::kSubdetOffset)) << srd::endl;
    }
    int id = thit->geographicalId().rawId();
    int subd =   thit->geographicalId().rawId() >> (DetId::kSubdetOffset);
    
    TrackerSingleRecHit const * hit= dynamic_cast<TrackerSingleRecHit const *>(thit);
    
    if (dynamic_cast<SiPixelRecHit const *>(hit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Pixel:" << subd << " ";
	verify(hit->omniCluster());
      }
    }
    if (dynamic_cast<SiStripRecHit1D const *>(hit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Strip1D:" << subd << " " << (id&3) << " ";
	verify(hit->omniCluster());
      }
    }
    if (dynamic_cast<SiStripRecHit2D const *>(hit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Strip2D:" << subd << " " << (id&3) << " ";
	verify(hit->omniCluster());
      }
    }
    if (dynamic_cast<SiStripMatchedRecHit2D const *>(thit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Strip Matched:" << subd << " " << (id&3) << " " << std::endl;
	// verify(hit->omniCluster());
      }
    }
    

  }
}

bool 
TrackerSingleRecHit::sharesInput( const TrackingRecHit* other, 
			      SharedInputType what) const
{
  verify(this); verify(other);
  // check subdetector is the same
  if( ((geographicalId().rawId()) >> (DetId::kSubdetOffset) ) != ( (other->geographicalId().rawId())>> (DetId::kSubdetOffset)) ) return false;
  //Protection against invalid hits
  if(!other->isValid()) return false;

  const std::type_info & otherType = typeid(*other);
  if (otherType == typeid(TrackerSingleRecHit)) {
    const TrackerSingleRecHit & otherCast = static_cast<const TrackerSingleRecHit&>(*other);
    return sharesInput(otherCast);
  } 

  if (otherType == typeid(ProjectedSiStripRecHit2D)) 
    return other->sharesInput(this,what);

  if (otherType == typeid(SiStripMatchedRecHit2D) ) {
    if (what == all) return false;
    return static_cast<SiStripMatchedRecHit2D const &>(*other).sharesInput(*this);
  }

  // last resort, recur to 'recHits()', even if it returns a vector by value
  std::vector<const TrackingRecHit*> otherHits = other->recHits();
  int ncomponents=otherHits.size();
  if(ncomponents==0)return false;
  if(ncomponents==1) return sharesInput(otherHits.front(),what);
  // ncomponents>1
  if(what == all )return false;

  for(int i=0;i<ncomponents;i++){
    if(sharesInput(otherHits[i],what))return true;
  }
  return false; 
}

// a simple hit has no components
std::vector<const TrackingRecHit*> TrackerSingleRecHit::recHits() const {
  std::vector<const TrackingRecHit*> nullvector;
  return nullvector; 
}
std::vector<TrackingRecHit*> TrackerSingleRecHit::recHits() {
  std::vector<TrackingRecHit*> nullvector;
  return nullvector; 
}

