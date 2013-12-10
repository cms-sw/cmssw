#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondTools/Ecal/interface/EcalTimeBiasCorrectionsXMLTranslator.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<EcalTimeBiasCorrections>: public  BaseValueExtractor<EcalTimeBiasCorrections> {
  public:

    typedef EcalTimeBiasCorrections Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what) {}
    void compute(Class const & it) override{}
  private:
  
  };

  template<>
  std::string PayLoadInspector<EcalTimeBiasCorrections>::dump() const {
    std::stringstream ss;
    EcalCondHeader h;
    ss << EcalTimeBiasCorrectionsXMLTranslator::dumpXML(h,object());
    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalTimeBiasCorrections>::summary() const {
    std::stringstream ss;
    object().print(ss);
    return ss.str();
  }

  template<>
  std::string PayLoadInspector<EcalTimeBiasCorrections>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }
}

PYTHON_WRAPPER(EcalTimeBiasCorrections,EcalTimeBiasCorrections);
