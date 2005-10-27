#include <DataFormats/CSCDigi/interface/CSCWireDigi.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <FWCore/EDProduct/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  namespace {

  CSCWireDigi d;
  CSCWireDigi::PersistentPacking bb;
  std::vector<CSCWireDigi>  vv;
  std::vector<std::vector<CSCWireDigi> >  v1; 
  CSCWireDigiCollection dd;
    
  edm::Wrapper<CSCWireDigiCollection> dw;

  }
}
