#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
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

namespace{ 
  namespace {
    DTRecHit1D hh;
    std::vector<DTRecHit1D>  vhh;
    LocalPoint lp;
    LocalVector lv;
    double c2;

    DTSLRecSegment2D dtSL;
    DTRecSegment2DCollection ds;

    edm::Wrapper<DTRecSegment2DCollection> dws;
  }
}


namespace{
  namespace {
    DTChamberRecSegment2D phi;
    DTRecSegment4D s4D;
    DTRecSegment4DCollection c4D;
    edm::Wrapper<DTRecSegment4DCollection> dws4D;
  }
}
