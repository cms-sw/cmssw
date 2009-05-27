#include "CondFormats/RunInfo/interface/HLTScaler.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<lumi::HLTScaler>: public  BaseValueExtractor<lumi::HLTScaler> {
  public:

    typedef lumi::HLTScaler Class;
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
  PayLoadInspector<lumi::HLTScaler>::dump() const {
    std::stringstream ss;
    object().print(ss);
    return ss.str();
  }
  
  template<>
  std::string PayLoadInspector<lumi::HLTScaler>::summary() const {
    std::stringstream ss;
    object().print(ss);
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<lumi::HLTScaler>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }
}

PYTHON_WRAPPER(lumi::HLTScaler,HLTScaler);
