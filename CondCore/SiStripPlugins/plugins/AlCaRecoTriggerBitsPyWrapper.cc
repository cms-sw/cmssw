
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include <boost/algorithm/string.hpp>
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<AlCaRecoTriggerBits>: public  BaseValueExtractor<AlCaRecoTriggerBits> {
  public:

    typedef AlCaRecoTriggerBits Class;
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
  PayLoadInspector<AlCaRecoTriggerBits>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<AlCaRecoTriggerBits>::summary() const {
    // std::stringstream ss;
	std::string result = "empty map";
	if( object().m_alcarecoToTrig.size() > 0 ) {
		//	ss << "trigger bit : value \n";
		//	std::map<std::string, std::string>::const_iterator it = object().m_alcarecoToTrig.begin();

		//std::vector<std::string> strs;
		//boost::split(strs, object().m_alcarecoToTrig.begin()->second, boost::is_any_of(";"));
		result = object().m_alcarecoToTrig.begin()->first +" :\n"+ object().m_alcarecoToTrig.begin()->second;
		boost::replace_all(result, ";", ";\n");
		//ss << it->first << " : " << it->second << "\n";
	}
	return result;
  }
  

  template<>
  std::string PayLoadInspector<AlCaRecoTriggerBits>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(AlCaRecoTriggerBits,AlCaRecoTriggerBits);
