#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  struct dictionary {

  RPCDigi d;
  std::vector<RPCDigi>  vv;
  std::vector<std::vector<RPCDigi> >  v1; 
  RPCDigiCollection dd;
    
  edm::Wrapper<RPCDigiCollection> dw;

  };
}
