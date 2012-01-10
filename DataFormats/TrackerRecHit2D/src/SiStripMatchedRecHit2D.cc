#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"



bool 
SiStripMatchedRecHit2D::sharesInput( const TrackingRecHit* other, 
				     SharedInputType what) const
{
  if (what==all && (geographicalId() != other->geographicalId())) return false;
 
  if (!sameDetModule(*other)) return false;
  
  if (!trackerHitRTTI::isMatched(*other) ) {
    if (what==all)  return false;
    return monoHit()->sharesInput( other,what) || stereoHit()->sharesInput( other,what);
  }
  
  const SiStripMatchedRecHit2D* otherMatched = static_cast<const SiStripMatchedRecHit2D*>(other);
  if ( what == all)
    return monoHit()->sharesInput(*otherMatched->monoHit()) && 
      stereoHit()->sharesInput(*otherMatched->stereoHit());

  return monoHit()->sharesInput(*otherMatched->monoHit()) || 
    stereoHit()->sharesInput(*otherMatched->stereoHit());
}



bool SiStripMatchedRecHit2D::sharesInput(TrackerSingleRecHit const & other) const {
  return (monoHit()->sharesInput( other)|| stereoHit()->sharesInput( other));
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


