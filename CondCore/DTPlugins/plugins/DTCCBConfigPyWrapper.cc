
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<DTCCBConfig>: public  BaseValueExtractor<DTCCBConfig> {
  public:

    typedef DTCCBConfig Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it){
    }
  private:
  
  };


  template<>
  std::string
  PayLoadInspector<DTCCBConfig>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<DTCCBConfig>::summary() const {
    std::stringstream ss;
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<DTCCBConfig>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(DTCCBConfig,DTCCBConfig);
