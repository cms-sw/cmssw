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
    virtual void fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result);
  private:
    std::string m_dirname;
    std::string m_filename;
    TFile* m_source;
  };
}//ns lumi
#endif
