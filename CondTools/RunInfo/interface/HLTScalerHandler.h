#ifndef CondTools_RunInfo_HLTScalerHandler_h
#define CondTools_RunInfo_HLTScalerHandler_h
#include <string>
#include "CondFormats/RunInfo/interface/HLTScaler.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class HLTScalerReaderBase;
  class HLTScalerHandler : public popcon::PopConSourceHandler<lumi::HLTScaler>{
  public:
    void getNewObjects();
    std::string id() const;
    ~HLTScalerHandler();
    explicit HLTScalerHandler(const edm::ParameterSet& pset); 
    
  private:
    std::string m_name;
    int m_startRun;
    int m_numberOfRuns;
    HLTScalerReaderBase* m_datareader;
  };
}//ns lumi
#endif
