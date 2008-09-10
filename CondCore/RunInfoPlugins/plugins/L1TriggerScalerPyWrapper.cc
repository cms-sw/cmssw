
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<L1TriggerScaler>: public  BaseValueExtractor<L1TriggerScaler> {
  public:

    typedef L1TriggerScaler Class;
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
  PayLoadInspector<L1TriggerScaler>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<L1TriggerScaler>::summary() const {
    std::stringstream ss;
    ss << object->m_run.back().m_rn;
    ss << ", "
    ss << object->m_run.back().m_lumisegment;
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<L1TriggerScaler>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(L1TriggerScaler,L1TriggerScaler);
