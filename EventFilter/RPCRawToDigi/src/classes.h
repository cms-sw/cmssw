#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRawSynchro.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace { 
    std::map< std::pair<int,int>, int > a1;
    edm::Wrapper<std::map< std::pair<int,int>, int > > a2;

    RPCRawDataCounts d1;
    edm::Wrapper<RPCRawDataCounts> d2;

    LinkBoardElectronicIndex e1;
    edm::Wrapper<LinkBoardElectronicIndex> e2;

    std::pair<LinkBoardElectronicIndex,int> f1;
    edm::Wrapper< std::pair<LinkBoardElectronicIndex,int> > f2;

    std::vector< std::pair<LinkBoardElectronicIndex, int> > g1;
    edm::Wrapper< std::vector< std::pair<LinkBoardElectronicIndex, int> > > g2;
  }
}
