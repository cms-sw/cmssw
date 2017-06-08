#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include "DataFormats/RPCDigi/interface/RPCRawDataCounts.h"
#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"
#include "DataFormats/RPCDigi/interface/RPCDigiL1Link.h"
#include "DataFormats/RPCDigi/interface/RPCAMCLinkCounters.h"

#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>
#include <map>

namespace DataFormats_RPCDigi {
  struct dictionary {
    
    RPCDigi d;
    std::vector<RPCDigi>  vv;
    std::vector<std::vector<RPCDigi> >  v1; 
    RPCDigiCollection dd;
    edm::Wrapper<RPCDigiCollection> dw;
    
    edm::Wrapper<std::map< std::pair<int,int>, int > > a2;
    
    RPCRawDataCounts d1;
    edm::Wrapper<RPCRawDataCounts> d2;
    
    LinkBoardElectronicIndex e1;
    edm::Wrapper<LinkBoardElectronicIndex> e2;
    
    std::pair<LinkBoardElectronicIndex,int> f1;
    edm::Wrapper< std::pair<LinkBoardElectronicIndex,int> > f2;
    
    std::vector< std::pair<LinkBoardElectronicIndex, int> > g1;
    edm::Wrapper< std::vector< std::pair<LinkBoardElectronicIndex, int> > > g2;
    
    std::vector<std::pair<unsigned int, int> > basic;
    edm::Wrapper<RPCDigiL1Link> l1w;
    std::vector<RPCDigiL1Link> plain;
    edm::Wrapper<std::vector<RPCDigiL1Link> > vectorplain;

    std::pair<unsigned int, std::uint32_t> puu;
    std::pair<std::pair<unsigned int, std::uint32_t>, unsigned int> ppuuu;
    std::map<std::pair<unsigned int, std::uint32_t>, unsigned int> mpuuu;
    edm::Wrapper<RPCAMCLinkCounters> ralc;
  };
}
