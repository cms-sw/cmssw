#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>
#include <map>

namespace{ 
  namespace {
    DTRecHit1D rh1d;
    DTRecHit1DPair p;
    DTRecHitCollection c;
    edm::Wrapper<DTRecHitCollection> w;
  }
}
