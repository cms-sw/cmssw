#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {
    std::pair<unsigned int, unsigned int> dummyrpc1;
    std::pair<unsigned long, unsigned long> dummyrpc2;
    std::map<RPCDetId, std::pair<unsigned int, unsigned int> > dummyrpcdetid1;
    std::map<RPCDetId, std::pair<unsigned long, unsigned long> > dummyrpcdetid2;

    RPCRecHit rrh;
    std::vector<RPCRecHit> vrh;
    RPCRecHitCollection c;
    edm::Wrapper<RPCRecHitCollection> w;
  };
}

