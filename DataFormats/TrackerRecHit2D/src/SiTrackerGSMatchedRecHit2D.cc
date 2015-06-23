#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

bool SiTrackerGSMatchedRecHit2D::sharesInput( const TrackingRecHit* other, 
					      SharedInputType what) const
{
  const SiTrackerGSMatchedRecHit2D * otherCasted = dynamic_cast<const SiTrackerGSMatchedRecHit2D*>(other);
  return bool(otherCasted) && otherCasted->id() == this->id() && otherCasted->eeId() == this->eeId();
}

