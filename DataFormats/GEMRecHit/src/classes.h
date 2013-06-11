#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {
    std::pair<unsigned int, unsigned int> dummyrpc1;
    std::pair<unsigned long, unsigned long> dummyrpc2;
    std::map<GEMDetId, std::pair<unsigned int, unsigned int> > dummyrpcdetid1;
    std::map<GEMDetId, std::pair<unsigned long, unsigned long> > dummyrpcdetid2;

    GEMRecHit rrh;
    std::vector<GEMRecHit> vrh;
    GEMRecHitCollection c;
    edm::Wrapper<GEMRecHitCollection> w;
  };
}

