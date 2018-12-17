#include "DataFormats/FTLRecHit/interface/FTLTrackingRecHit.h"

template<>
bool FTLTrackingRecHit<FTLRecHitRef>::sharesInput( const TrackingRecHit* other, SharedInputType what) const 
{
    if (typeid(*other) == typeid(FTLTrackingRecHit<FTLRecHitRef>)) {
        return objRef() == (static_cast<const FTLTrackingRecHit<FTLRecHitRef> *>(other))->objRef();
    } else {
        return false;
    }
}
