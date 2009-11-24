#ifndef CondTools_Luminosity_MixedSource_h
#define CondTools_Luminosity_MixedSource_h
#include <vector>
#include <string>
#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
class TFile;
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class LumiSectionData;
  /**This source takes lumi measurement data from Lumi Root file and L1 data from L1 database 
   **/
  class MixedSource:public LumiRetrieverBase{
  public:
    explicit MixedSource(const edm::ParameterSet& pset);
    virtual ~MixedSource(){}
    virtual const std::string 
      fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result, bool allowForceFirstSince=false);
  private:
    std::string m_filename;
    TFile* m_source;
    std::string m_lumiversion;
    std::string m_trgdb;
  };
}//ns lumi
#endif
