#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/DTDigi/interface/DTLocalTrigger.h>
#include <DataFormats/DTDigi/interface/DTLocalTriggerCollection.h>
#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  namespace {

  DTDigi d;
  std::vector<DTDigi>  vv;
  std::vector<std::vector<DTDigi> >  v1; 
  DTDigiCollection dd;

  DTLocalTrigger t;
  std::vector<DTLocalTrigger>  ww;
  std::vector<std::vector<DTLocalTrigger> >  w1; 
  DTLocalTriggerCollection tt;
    
  edm::Wrapper<DTDigiCollection> dw;
  edm::Wrapper<DTLocalTriggerCollection> tw;

  }
}
