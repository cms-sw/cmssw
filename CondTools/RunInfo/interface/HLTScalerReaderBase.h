#ifndef CondTools_RunInfo_HLTScalerReaderBase_h
#define CondTools_RunInfo_HLTScalerReaderBase_h
#include <vector>
#include "CondCore/DBCommon/interface/Time.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lumi{
  class HLTScaler;
  class HLTScalerReaderBase{
  public:
    explicit HLTScalerReaderBase(const edm::ParameterSet& pset):m_pset(pset){}
    virtual ~HLTScalerReaderBase(){}
    virtual void fill(int startRun,int numberOfRuns,std::vector< std::pair<lumi::HLTScaler*,cond::Time_t> >& result)=0;
  protected:
    const edm::ParameterSet& m_pset;
  };
}//ns lumi
#endif
