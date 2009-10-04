#ifndef CondTools_Luminosity_LumiSectionDataHandler_h
#define CondTools_Luminosity_LumiSectionDataHandler_h
#include <string>
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class LumiRetrieverBase;
  class LumiSectionDataHandler : public popcon::PopConSourceHandler<lumi::LumiSectionData>{
  public:
    explicit LumiSectionDataHandler(const edm::ParameterSet& pset); 
    void getNewObjects();
    std::string id() const;
    ~LumiSectionDataHandler();
  private:
    std::string m_name;
    int m_runnumber;
    LumiRetrieverBase* m_datareader;
    short m_lumiversionnumber;
  };
}//ns lumi
#endif
