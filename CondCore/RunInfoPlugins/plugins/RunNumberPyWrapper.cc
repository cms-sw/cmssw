
#include "CondFormats/RunInfo/interface/RunNumber.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

using namespace runinfo_test;

namespace cond {

  template<>
  class ValueExtractor<RunNumber>: public  BaseValueExtractor<RunNumber> {
  public:

    typedef RunNumber Class;
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
  PayLoadInspector<RunNumber>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<RunNumber>::summary() const {
    std::stringstream ss;
    ss << object().m_runnumber.size() <<", ";
    if (!object().m_runnumber.empty()) {
      ss << object().m_runnumber.front().m_run;
      ss << ", ";
      ss << object().m_runnumber.front().m_name;
      ss << ", " << object().m_runnumber.front().m_start_time_str;
      ss << "; ";
      ss << ", " << object().m_runnumber.front().m_stop_time_str;
      ss << "; ";
      for (size_t i=0; i< object().m_runnumber.front().m_subdt_joined.size(); ++i){
      ss << ", " << object().m_runnumber.front().m_subdt_joined[i];
      ss << "; ";
      }
     
    }
    return ss.str();
   
  }
  

  template<>
  std::string PayLoadInspector<RunNumber>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(RunNumber,RunNumber);
