
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<RunInfo>: public  BaseValueExtractor<RunInfo> {
  public:

    typedef RunInfo Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it){
      this->add(it.m_start_current);
      this->add(it.m_stop_current);
      this->add(it.m_avg_current);
      this->add(it.m_max_current);
      this->add(it.m_min_current);
      this->add(it.m_run_intervall_micros);
    }
  private:
  
  };


  template<>
  std::string
  PayLoadInspector<RunInfo>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<RunInfo>::summary() const {
    std::stringstream ss;
    ss << "RUN: "<<object().m_run;
    ss << ", START TIME: " << object().m_start_time_str;
    ss << ", STOP TIME:" << object().m_stop_time_str;
    ss << ", START CURRENT:  " << object().m_start_current;
    ss << ", STOP CURRENT:  " << object().m_stop_current;
    ss << ", AVG CURRENT:  " << object().m_avg_current;
    ss << ", MIN CURRENT:  " << object().m_min_current;
    ss << ", MAX CURRENT:  " << object().m_max_current;
    ss << ", RUN INTERVALL IN MICROSECONDS: "<< object().m_run_intervall_micros;  /*
     ss << ", ALL CURRENT VALUE FROM STOP TO START (BACKWARD) :" ;
    for (size_t i=0; i< object().m_current.size() ; i++){
      ss<< object().m_current[i] << ", ";
    } 
										 */    
    ss << ", FED IN :" ;
    for (size_t i=0; i<object().m_fed_in.size(); i++){
      ss<< object().m_fed_in[i] << ", "; 
    }  
      
  return ss.str();
}
  

  template<>
  std::string PayLoadInspector<RunInfo>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(RunInfo,RunInfo);
