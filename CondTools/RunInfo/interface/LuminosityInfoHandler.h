#ifndef CondTools_RunInfo_LuminosityInfoHandler_h
#define CondTools_RunInfo_LuminosityInfoHandler_h
#include <string>
#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class LumiReaderBase;
  class LuminosityInfoHandler : public popcon::PopConSourceHandler<lumi::LuminosityInfo>{
  public:
    void getNewObjects();
    std::string id() const;
    ~LuminosityInfoHandler();
    explicit LuminosityInfoHandler(const edm::ParameterSet& pset); 
    
  private:
    std::string m_name;
    int m_startRun;
    int m_numberOfRuns;
    LumiReaderBase* m_datareader;
  };
}//ns lumi
#endif
