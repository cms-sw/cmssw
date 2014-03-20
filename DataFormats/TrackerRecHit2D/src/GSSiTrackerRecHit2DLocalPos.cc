#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

namespace {

  const OmniClusterRef nullRef;

}

OmniClusterRef const & GSSiTrackerRecHit2DLocalPos::firstClusterRef() const { return nullRef;}

