#ifndef CondTools_Luminosity_NoDataSource_h
#define CondTools_Luminosity_NoDataSource_h
#include <vector>
#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class LumiSectionData;
  class NoDataSource:public LumiRetrieverBase{
  public:
    explicit NoDataSource(const edm::ParameterSet& pset);
    virtual ~NoDataSource(){}
    //return a operation comment string to be injected in the log file 
    const std::string fill(std::vector< std::pair< lumi::LumiSectionData*,cond::Time_t > >& result, bool allowForceFirstSince=false);
  };
}//ns lumi
#endif
