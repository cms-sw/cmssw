#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace { 
    std::pair<int, std::vector<int> > b1;
    edm::Wrapper<std::pair<int, std::vector<int> > > b2;
    std::map<int, std::vector<int> > c1;
    edm::Wrapper<std::map<int, std::vector<int> > > c2;
    RPCRawDataCounts d1;
    edm::Wrapper<RPCRawDataCounts> d2;
  }
}
