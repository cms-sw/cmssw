#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<lumi::LuminosityInfo>: public  BaseValueExtractor<lumi::LuminosityInfo> {
  public:

    typedef lumi::LuminosityInfo Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
    {

    }
    void compute(Class const & it){
    }
  private:
  
  };


  template<>
  std::string
  PayLoadInspector<lumi::LuminosityInfo>::dump() const {
    std::stringstream ss;
    object().print(ss);
    return ss.str();
  }
  
  template<>
  std::string 
  PayLoadInspector<lumi::LuminosityInfo>::summary() const {
    std::stringstream ss;
    object().print(ss);
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<lumi::LuminosityInfo>::plot(std::string const & filename,std::string const &, std::vector<int> const&, std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }
}

PYTHON_WRAPPER(lumi::LuminosityInfo,LuminosityInfo);
