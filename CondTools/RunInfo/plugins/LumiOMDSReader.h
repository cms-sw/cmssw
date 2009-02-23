#ifndef CondTools_Luminosity_LumiOMDSReader_h
#define CondTools_Luminosity_LumiOMDSReader_h
#include <vector>
#include <string>
#include "CondTools/RunInfo/interface/LumiReaderBase.h"
namespace edm{
  class ParameterSet;
}
namespace cond{
  class DBSession;
}
namespace lumi{
  class LuminosityInfo;
  class LumiOMDSReader:public LumiReaderBase{
  public:
    explicit LumiOMDSReader(const edm::ParameterSet& pset);
    virtual ~LumiOMDSReader();
    virtual void fill(int startRun, int numberOfRuns, 
		      std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >& result, short lumiversionid);
  private:
    cond::DBSession* m_session;
    std::string m_constr;
  };
}//ns lumi
#endif
