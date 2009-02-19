#ifndef CondTools_RunInfo_HLTScalerDBReader_h
#define CondTools_RunInfo_HLTSCalerDBReader_h
#include <vector>
#include <string>
#include "CondTools/RunInfo/interface/HLTScalerReaderBase.h"
namespace edm{
  class ParameterSet;
}
namespace cond{
  class DBSession;
}
namespace lumi{
  class HLTScaler;
  class HLTScalerDBReader:public HLTScalerReaderBase{
  public:
    explicit HLTScalerDBReader(const edm::ParameterSet& pset);
    virtual ~HLTScalerDBReader();
    virtual void fill(int startRun, int numberOfRuns, 
		      std::vector< std::pair<lumi::HLTScaler*,cond::Time_t> >& result);
  private:
    cond::DBSession* m_session;
    std::string m_constr;
  };
}//ns lumi
#endif
