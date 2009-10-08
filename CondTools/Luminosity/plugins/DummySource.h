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
    //return a operation comment string to be injected in the log file 
    const std::string fill(std::vector< std::pair< lumi::LumiSectionData*,cond::Time_t > >& result, bool allowForceFirstSince=false);
  private:
    std::string m_lumiversion;
    size_t m_runnumber;
  };
}//ns lumi
#endif
