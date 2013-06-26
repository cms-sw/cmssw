#ifndef _TrackHitsFilter_h_
#define _TrackHitsFilter_h_

namespace reco { class Track; }

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include <vector>

class TrackHitsFilter
{
  public:
    virtual ~TrackHitsFilter() {}
    virtual bool operator()
      (const reco::Track*, std::vector<const TrackingRecHit *>) const = 0;
};

#endif
