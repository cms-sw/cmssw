#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>
#include <map>

namespace{ 
  namespace {
    DTRecHit1DPair p;
    std::vector<DTRecHit1DPair>  vv;
    std::vector<std::vector<DTRecHit1DPair> >  v1; 
    DTRecHitCollection dd;
    edm::Wrapper<DTRecHitCollection> dw;
  }
}
