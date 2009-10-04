#ifndef CondTools_Luminosity_DummySource_h
#define CondTools_Luminosity_DummySource_h
#include <vector>
#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class LumiSectionData;
  class DummySource:public LumiRetrieverBase{
  public:
    explicit DummySource(const edm::ParameterSet& pset):LumiRetrieverBase(pset){}
    virtual ~DummySource(){}
    void fill(int runnumber, std::vector< std::pair< lumi::LumiSectionData*,cond::Time_t > >& result, short lumiversionid);
  };
}//ns lumi
#endif
