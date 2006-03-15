#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  namespace {

  DTDigi d;
  DTDigi::PersistentPacking bb;
  std::vector<DTDigi>  vv;
  std::vector<std::vector<DTDigi> >  v1; 
  DTDigiCollection dd;
    
  edm::Wrapper<DTDigiCollection> dw;

  }
}
