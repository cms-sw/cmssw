#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>
#include <map>

namespace{ 
  namespace {
    DTRecHit1D rh1d;
    DTRecHit1DPair p;

    std::pair<const DTRecHit1D*, const DTRecHit1D*> pair;
    std::vector<DTRecHit1DPair>  vv;
    std::vector<std::vector<DTRecHit1DPair> >  v1;
    
    std::map<DTLayerId,std::vector<DTRecHit1DPair> > map1;
    edm::ClonePolicy<DTRecHit1DPair> clp;
    edm::OwnVector<DTRecHit1DPair,edm::ClonePolicy<DTRecHit1DPair> > ov;
    DTRecHitCollection dd;
    edm::Wrapper<DTRecHitCollection> dw;
  }
}
