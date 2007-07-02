#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"


SiStripMatchedRecHit2D::SiStripMatchedRecHit2D( const LocalPoint& pos, const LocalError& err,
								const DetId& id , const SiStripRecHit2D* rMono,const SiStripRecHit2D* rStereo): BaseSiTrackerRecHit2DLocalPos(pos, err, id), componentMono_(*rMono),componentStereo_(*rStereo){}

bool 
SiStripMatchedRecHit2D::sharesInput( const TrackingRecHit* other, 
				     SharedInputType what) const
{
  if (trackerId() != other->geographicalId()) return false;
  
  const SiStripMatchedRecHit2D* otherMatched = 
    dynamic_cast<const SiStripMatchedRecHit2D*>(other);
  if ( otherMatched == 0 )  return false;

  if ( what == all) {
    return (monoHit()->sharesInput( otherMatched->monoHit(),what) && 
	    stereoHit()->sharesInput( otherMatched->stereoHit(),what));
  }
  else {
    return (monoHit()->sharesInput( otherMatched->monoHit(),what) || 
	    stereoHit()->sharesInput( otherMatched->stereoHit(),what));
  }
}
