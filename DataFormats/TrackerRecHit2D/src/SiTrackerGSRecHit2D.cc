#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"


bool SiTrackerGSRecHit2D::sharesInput( const TrackingRecHit* other, 
					      SharedInputType what) const
{
  const SiTrackerGSRecHit2D * otherCasted = dynamic_cast<const SiTrackerGSRecHit2D*>(other);
  return bool(otherCasted) && otherCasted->id() == this->id() && otherCasted->eeId() == this->eeId();
}
