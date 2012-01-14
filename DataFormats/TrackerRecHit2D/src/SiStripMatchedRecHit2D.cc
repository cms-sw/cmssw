#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"



bool 
SiStripMatchedRecHit2D::sharesInput( const TrackingRecHit* other, 
				     SharedInputType what) const
{
  if (what==all && (geographicalId() != other->geographicalId())) return false;
 
  if (!sameDetModule(*other)) return false;

  if (trackerHitRTTI::isMatched(*other) ) {
    const SiStripMatchedRecHit2D* otherMatched = static_cast<const SiStripMatchedRecHit2D*>(other);
    return sharesClusters(*this, *otherMatched,what);
  }
   
  if (what==all)  return false;
  // what about multi???
  auto const & otherClus = reinterpret_cast<const BaseTrackerRecHit *>(hit)->firstClusterRef();
  return (otherClus==stereoClusterRef())  ||  (otherClus==monoClusterRef());
  
  
}



bool SiStripMatchedRecHit2D::sharesInput(TrackerSingleRecHit const & other) const {
  return other.sameCluster(monoClusterRef()) || other.sameCluster(stereoClusterRef());
}


std::vector<const TrackingRecHit*>
SiStripMatchedRecHit2D::recHits()const {
  std::vector<const TrackingRecHit*> rechits(2);
  rechits[0]=&componentMono_;
  rechits[1]=&componentStereo_;
  return rechits;
}

std::vector<TrackingRecHit*>
SiStripMatchedRecHit2D::recHits() {
  std::vector<TrackingRecHit*> rechits(2);
  rechits[0]=&componentMono_;
  rechits[1]=&componentStereo_;
  return rechits;
}


