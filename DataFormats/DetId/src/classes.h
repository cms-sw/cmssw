#include "DataFormats/DetId/interface/DetId.h"
#include <boost/cstdint.hpp> 
#include <map>
#include <vector>

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    std::vector<DetId> dummy;
    edm::EDCollection<DetId> vDI_;
    DetIdCollection theDI_;
    edm::Wrapper<DetIdCollection> anotherDIw_;
    edm::Wrapper< edm::EDCollection<DetId> > theDIw_;
    std::vector<std::pair<DetId,float> > dummyPairFloat;
    std::pair<DetId,float>    thepair;           
    std::map<DetId, std::pair<unsigned int, unsigned int> > dummytrkrechit2d1;
    std::map<DetId, std::pair<unsigned long, unsigned long> > dummytrkrechit2d2;
  };
}
