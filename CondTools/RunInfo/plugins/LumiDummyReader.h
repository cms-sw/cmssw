#ifndef CondTools_RunInfo_LumiDummyReader_h
#define CondTools_RunInfo_LumiDummyReader_h
#include <vector>
#include "CondTools/RunInfo/interface/LumiReaderBase.h"
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class LuminosityInfo;
  class LumiDummyReader:public LumiReaderBase{
  public:
    explicit LumiDummyReader(const edm::ParameterSet& pset):LumiReaderBase(pset){}
    virtual ~LumiDummyReader(){}
    void fill(int startRun, int numberOfRuns, 
	      std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >& result, short lumiversionid);
  };
}//ns lumi
#endif
