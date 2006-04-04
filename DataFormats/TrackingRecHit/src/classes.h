#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"
#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<TrackingRecHit*> v1;
    TrackingRecHitCollection c1;
    TrackingRecHitRef r1;
    InvalidTrackingRecHit i;
    TrackingRecHitRefProd rp1;
    TrackingRecHitRefVector rv1;
    edm::Wrapper<TrackingRecHitCollection> w1;
  }
}
