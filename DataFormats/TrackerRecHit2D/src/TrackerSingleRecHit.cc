#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"



#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

bool 
TrackerSingleRecHit::sharesInput( const TrackingRecHit* other, 
			      SharedInputType what) const
{
  //here we exclude non si-strip subdetectors
  if( ((geographicalId().rawId()) >> (DetId::kSubdetOffset) ) != ( (other->geographicalId().rawId())>> (DetId::kSubdetOffset)) ) return false;

  //Protection against invalid hits
  if(!other->isValid()) return false;

  const std::type_info & otherType = typeid(*other);
  if (otherType == typeid(TrackerSingleRecHit)) {
    const TrackerSingleRecHit & otherCast = static_cast<const TrackerSingleRecHit&>(*other);
    return sharesInput(otherCast);
  } else if (otherType == typeid(ProjectedSiStripRecHit2D)) {
    other.sharesInput(this,what);
  } else if (otherType == typeid(SiStripMatchedRecHit2D) ) {
    if (what == all) return false;
    static_cast<SiStripMatchedRecHit2D const &>(*other).sharesInput(*this);
  } else {
    // last resort, recur to 'recHits()', even if it returns a vector by value
    std::vector<const TrackingRecHit*> otherHits = other->recHits();
    int ncomponents=otherHits.size();
    if(ncomponents==0)return false;
    else if(ncomponents==1)return sharesInput(otherHits.front(),what);
    else if (ncomponents>1){
      if(what == all )return false;
      else{
	for(int i=0;i<ncomponents;i++){
	  if(sharesInput(otherHits[i],what))return true;
	}
	return false;
      }
    }
    return false;
  }
}

// a simple hit has no components
std::vector<const TrackingRecHit*> SiStripRecHit1D::recHits() const {
  std::vector<const TrackingRecHit*> nullvector;
  return nullvector; 
}
std::vector<TrackingRecHit*> SiStripRecHit1D::recHits() {
  std::vector<TrackingRecHit*> nullvector;
  return nullvector; 
}

