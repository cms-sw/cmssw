
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<SiStripBadStrip>: public  BaseValueExtractor<SiStripBadStrip> {
  public:

    typedef SiStripBadStrip Class;
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
  PayLoadInspector<SiStripBadStrip>::dump() const {
    std::stringstream ss;
    object().printDebug(ss);
    return ss.str();    
  }
  
  template<>
  std::string PayLoadInspector<SiStripBadStrip>::summary() const {
    std::stringstream ss;
    object().printSummary(ss);
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<SiStripBadStrip>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(SiStripBadStrip,SiStripBadStrip);
