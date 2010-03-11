#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"
#include "DataFormats/DTRecHit/interface/DTRecClusterCollection.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>
#include <map>

namespace{
  struct dictionary {
    std::map<DTLayerId,std::pair<unsigned int,unsigned int> > dummydtlayerid1;
    std::map<DTLayerId,std::pair<unsigned long,unsigned long> > dummydtlayerid2;
    std::map<DTSuperLayerId,std::pair<unsigned int,unsigned int> > dummyslayerid1;
    std::map<DTSuperLayerId,std::pair<unsigned long,unsigned long> > dummyslayerid2;
    std::map<DTChamberId,std::pair<unsigned int,unsigned int> > dummychamberid1;
    std::map<DTChamberId,std::pair<unsigned long,unsigned long> > dummychamberid2;

    DTRecHit1D rh1d;
    DTRecHit1DPair p;
    DTRecHitCollection c;
    edm::Wrapper<DTRecHitCollection> w;
    DTRecHit1DPair hhh;
    std::vector<DTRecHit1DPair>  vhhh;
    LocalPoint lpp;
    LocalError lee;
    double c23;

    DTSLRecCluster dtCl;
    DTRecClusterCollection dc;

    edm::Wrapper<DTRecClusterCollection> dwc;

    DTRecHit1D hh;
    std::vector<DTRecHit1D>  vhh;
    LocalPoint lp;
    LocalVector lv;
    double c2;

    DTSLRecSegment2D dtSL;
    DTRecSegment2DCollection ds;

    edm::Wrapper<DTRecSegment2DCollection> dws;

    DTChamberRecSegment2D phi;
    DTRecSegment4D s4D;
    DTRecSegment4DCollection c4D;
    edm::Wrapper<DTRecSegment4DCollection> dws4D;

    DTRecSegment4DRef ref4D;
  };
}
