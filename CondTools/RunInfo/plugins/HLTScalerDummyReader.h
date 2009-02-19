#ifndef CondTools_RunInfo_HLTScalerDummyReader_h
#define CondTools_RunInfo_HLTScalerDummyReader_h
#include <vector>
#include "CondTools/RunInfo/interface/HLTScalerReaderBase.h"
namespace edm{
  class::ParameterSet;
}
namespace lumi{
  class HLTScaler;
  class HLTScalerDummyReader:public HLTScalerReaderBase{
  public:
    explicit HLTScalerDummyReader(const edm::ParameterSet& pset):HLTScalerReaderBase(pset){}
    virtual ~HLTScalerDummyReader(){}
    void fill(int startRun, int numberOfRuns, 
          std::vector< std::pair<lumi::HLTScaler*,cond::Time_t> >& result);
  };
}//ns lumi
#endif
