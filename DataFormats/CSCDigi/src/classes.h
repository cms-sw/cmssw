#include <DataFormats/CSCDigi/interface/CSCWireDigi.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigi.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h>
#include <FWCore/EDProduct/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  namespace {

  CSCWireDigi cWD_;
  CSCRPCDigi  cRD_;

  CSCWireDigi::PersistentPacking ppWD_;
  CSCRPCDigi::PersistentPacking ppRD_;

  std::vector<CSCWireDigi>  vWD_;
  std::vector<CSCRPCDigi>   vRD_;

  std::vector<std::vector<CSCWireDigi> >  vvWD_; 
  std::vector<std::vector<CSCRPCDigi>  >  vvRD_;

  CSCWireDigiCollection clWD_;
  CSCRPCDigiCollection  clRD_;
    
  edm::Wrapper<CSCWireDigiCollection> wWD_;
  edm::Wrapper<CSCRPCDigiCollection> wRD_;

  }
}
