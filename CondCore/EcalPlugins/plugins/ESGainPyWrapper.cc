#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondTools/Ecal/interface/ESGainXMLTranslator.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<ESGain>: public  BaseValueExtractor<ESGain> {
  public:

    typedef ESGain Class;
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
  PayLoadInspector<ESGain>::dump() const {
    std::stringstream ss;
    EcalCondHeader h;
    ss<<ESGainXMLTranslator::dumpXML(h,object());
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<ESGain>::summary() const {
    std::stringstream ss;
    object().print(ss);
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<ESGain>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(ESGain,ESGain);
