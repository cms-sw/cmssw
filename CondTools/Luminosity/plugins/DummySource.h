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
    explicit DummySource(const edm::ParameterSet& pset);
    virtual ~DummySource(){}
    void fill(std::vector< std::pair< lumi::LumiSectionData*,cond::Time_t > >& result);
  private:
    size_t m_lumiversion;
    size_t m_runnumber;
  };
}//ns lumi
#endif
