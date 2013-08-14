
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondTools/Ecal/interface/EcalTBWeightsXMLTranslator.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<EcalTBWeights>: public  BaseValueExtractor<EcalTBWeights> {
  public:

    typedef EcalTBWeights Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it) override{
    }
  private:
  
  };


  template<>
  std::string
  PayLoadInspector<EcalTBWeights>::dump() const {

    std::stringstream ss; 
    EcalCondHeader h;
    ss<< EcalTBWeightsXMLTranslator::dumpXML(h,object());
    return ss.str();
   
    
  }
  
  template<>
  std::string PayLoadInspector<EcalTBWeights>::summary() const {
    std::stringstream ss; 
    EcalCondHeader h;
    ss<< EcalTBWeightsXMLTranslator::dumpXML(h,object());
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<EcalTBWeights>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(EcalTBWeights,EcalTBWeights);
