#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include <iostream>
#include <typeinfo>
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

//#define DO_INTERNAL_CHECKS
#if defined(DO_INTERNAL_CHECKS)
namespace {
  
  void verify(OmniClusterRef const ref) {
    std::cout << 
      ref.rawIndex() << " " <<
      ref.isValid() << " " <<
      ref.isPixel() << " " <<
      ref.isStrip()  << " " <<
//      ref.isRegional() << " " <<
      ref.cluster_strip().isNull() << " " <<
      ref.cluster_pixel().isNull()  << " " << std::endl;
  }
  
  void verify(TrackingRecHit const * thit) {

    static bool once=true;
    if (once) {
      once=false;
      DetId Lim(DetId::Tracker,3);
      std::cout << "detid lim " << (Lim.rawId() >> (DetId::kSubdetOffset)) << std::endl;
    }
    int id = thit->geographicalId().rawId();
    int subd =   thit->geographicalId().rawId() >> (DetId::kSubdetOffset);
    
    TrackerSingleRecHit const * hit= dynamic_cast<TrackerSingleRecHit const *>(thit);
    BaseTrackerRecHit const * bhit = dynamic_cast<BaseTrackerRecHit const *>(thit);    
    if (!bhit)
      std::cout << "not a tracker hit! " << typeid(*thit).name() << std::endl;

    if (trackerHitRTTI::isUndef(*thit))
      std::cout << "undef hit! " << typeid(*thit).name() << std::endl;
    
    if (dynamic_cast<SiPixelRecHit const *>(hit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Pixel:" << subd << " " << bhit->isSingle() << ". ";
	verify(hit->omniCluster());
      }
    }
    if (dynamic_cast<SiStripRecHit1D const *>(hit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Strip1D:" << subd << " " << (id&3) << " "<< bhit->isSingle() << ". ";
	verify(hit->omniCluster());
      }
    }
    if (dynamic_cast<SiStripRecHit2D const *>(hit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Strip2D:" << subd << " " << (id&3) << " "<< bhit->isSingle() << ". ";
	verify(hit->omniCluster());
      }
    }
    if (dynamic_cast<SiStripMatchedRecHit2D const *>(thit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Strip Matched:" << subd << " " << (id&3) << " " << bhit->isMatched() << ". " << std::endl;
	// verify(hit->omniCluster());
      }
    }
    if (dynamic_cast<ProjectedSiStripRecHit2D const *>(thit)) {
      static int n=0;
      if (++n<5) {
	std::cout << "Strip Matched:" << subd << " " << (id&3) << " " << bhit->isProjected() << ". " << std::endl;
	// verify(hit->omniCluster());
      }
    }
    

  }
  
  void problem(const TrackingRecHit* thit, const char * what) {
    std::cout << "not sharing with itself! " << what << " "
	      << typeid(*thit).name() << std::endl;
    verify(thit);

  }

  bool doingCheck = false;
  inline void checkSelf(const TrackingRecHit* one,const TrackingRecHit* two) {
    doingCheck=true;
    if (!one->sharesInput(one,TrackingRecHit::all)) problem(one,"all");
    if (!one->sharesInput(one,TrackingRecHit::some)) problem(one,"some");
    if (!two->sharesInput(two,TrackingRecHit::all)) problem(two,"all");
    if (!two->sharesInput(two,TrackingRecHit::some)) problem(two,"some");
    doingCheck=false;
  }
}
#endif

bool 
TrackerSingleRecHit::sharesInput( const TrackingRecHit* other, 
			      SharedInputType what) const
{
#if defined(DO_INTERNAL_CHECKS)
  verify(this); verify(other);
  if (!doingCheck && (other!=this)) checkSelf(this,other);
#endif

  if (!sameDetModule(*other)) return false;

  // move to switch?
  if (trackerHitRTTI::isSingleType(*other)) {
    const TrackerSingleRecHit & otherCast = static_cast<const TrackerSingleRecHit&>(*other);
    return sharesInput(otherCast);
  } 

  if (trackerHitRTTI::isMatched(*other) ) {
    if (what == all) return false;
    return static_cast<SiStripMatchedRecHit2D const &>(*other).sharesInput(*this);
  }

  // last resort, recur to 'recHits()', even if it returns a vector by value
  std::vector<const TrackingRecHit*> otherHits = other->recHits();
  int ncomponents=otherHits.size();
  if(ncomponents==0)return false; //bho
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

