#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"
#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include <vector>
#include "DataFormats/Common/interface/ClonePolicy.h"

namespace {
  namespace {
    std::vector<TrackingRecHit*> v1;
    edm::OwnVector<TrackingRecHit,
      edm::ClonePolicy<TrackingRecHit> > a6;
  }
}
