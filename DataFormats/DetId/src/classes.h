#include "DataFormats/DetId/interface/DetId.h"
#include <boost/cstdint.hpp> 
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
  };
}
