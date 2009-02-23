#ifndef CondTools_RunInfo_LumiReaderBase_h
#define CondTools_RunInfo_LumiReaderBase_h
#include <vector>
#include "CondCore/DBCommon/interface/Time.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lumi{
  class LuminosityInfo;
  class LumiReaderBase{
  public:
    explicit LumiReaderBase(const edm::ParameterSet& pset):m_pset(pset){}
    virtual ~LumiReaderBase(){}
    virtual void fill(int startRun,int numberOfRuns,std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >& result, short lumiversionid)=0;
  protected:
    const edm::ParameterSet& m_pset;
  };
}//ns lumi
#endif
