
#include "CondFormats/RunInfo/interface/RunSummary.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<RunSummary>: public  BaseValueExtractor<RunSummary> {
  public:

    typedef RunSummary Class;
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
  PayLoadInspector<RunSummary>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<RunSummary>::summary() const {
    std::stringstream ss;
      ss << "RUN: "<<object().m_run;
      ss << ", NAME: " <<object().m_name;
      ss << ", START TIME: " << object().m_start_time_str;
      ss << ", STOP TIME:" << object().m_stop_time_str;
      ss << ", LUMISECTIONS: " << object().m_lumisections;
      ss << ", HLT_KEY:  " << object().m_hltkey;
      ss << ", TRG_NEVENTS:  " << object().m_nevents;
      ss << ", TRG_RATE:  " << object().m_rate;
      ss << ", SUBDETS IB RUN: ";  
      for (size_t i=0; i<object().m_subdt_in.size() ; i++){
	if (object().m_subdt_in[i]==0) {
	  ss<< "PIXEL, ";
	}
	if (object().m_subdt_in[i]==1) {
	  ss<<"TRACKER, ";
	}
	if (object().m_subdt_in[i]==2) {
	  ss<< "ECAL, " ;
	}
	if (object().m_subdt_in[i]==3) {
	  ss<< "HCAL, ";
	}
	if (object().m_subdt_in[i]==4) {
	  ss<<"DT," ;
	}  
	if (object().m_subdt_in[i]==5) {
	  ss<<"CSC,";
	}   
	if (object().m_subdt_in[i]==6) {
	  ss<<"RPC, ";
	}  
      }
      
    
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<RunSummary>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(RunSummary,RunSummary);
