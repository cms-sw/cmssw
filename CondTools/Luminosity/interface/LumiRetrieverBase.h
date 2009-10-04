#ifndef CondTools_Luminosity_LumiRetrieverBase_h
#define CondTools_Luminosity_LumiRetrieverBase_h
#include <vector>
#include "CondCore/DBCommon/interface/Time.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lumi{
  class LumiSectionData;
  class LumiRetrieverBase{
  public:
    explicit LumiRetrieverBase(const edm::ParameterSet& pset):m_pset(pset){}
    virtual ~LumiRetrieverBase(){}
    virtual void fill(int runnumber,std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t > >& result, short lumiversionid)=0;
  protected:
    const edm::ParameterSet& m_pset;
  };
}//ns lumi
#endif
