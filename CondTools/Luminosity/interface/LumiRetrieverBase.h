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
    virtual void fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t > >& result)=0;
    edm::ParameterSetID parametersetId() const;
  protected:
    const edm::ParameterSet& m_pset;
  };
}//ns lumi
#endif
