#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

namespace {
    const OmniClusterRef nullRef;
}

OmniClusterRef const & FastTrackerRecHit::firstClusterRef() const { return nullRef;}

