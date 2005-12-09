#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <FWCore/EDProduct/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  namespace {

  RPCDigi d;
  RPCDigi::PersistentPacking bb;
  std::vector<RPCDigi>  vv;
  std::vector<std::vector<RPCDigi> >  v1; 
  RPCDigiCollection dd;
    
  edm::Wrapper<RPCDigiCollection> dw;

  }
}
