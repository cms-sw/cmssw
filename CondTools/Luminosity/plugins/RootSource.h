#ifndef CondTools_Luminosity_RootSource_h
#define CondTools_Luminosity_RootSource_h
#include <vector>
#include <string>
#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
class TFile;
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class LumiSectionData;
  class RootSource:public LumiRetrieverBase{
  public:
    explicit RootSource(const edm::ParameterSet& pset);
    virtual ~RootSource(){}
    virtual const std::string 
      fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result, bool allowForceFirstSince=false);
  private:
    std::string m_filename;
    TFile* m_source;
    std::string m_lumiversion;
  };
}//ns lumi
#endif
