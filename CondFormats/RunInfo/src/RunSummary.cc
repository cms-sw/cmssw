#include "CondFormats/RunInfo/interface/RunSummary.h"
//#include <sstream>

RunSummary::RunSummary():
	m_run(0)
	,m_sequenceName("null")
	,m_globalConfKey("null")
	,m_start_time_ll(0)
	,m_stop_time_ll(0)
	,m_start_time_str("null")
	,m_stop_time_str("null")
	,m_fill(-1)
	,m_energy(-1)
	,m_lumisections(0)
	,m_HLTKey(0)
	,m_HLTKeyDesc("null")
	,m_eventNumber(0)
	,m_triggerNumber(0)
	,m_avgTriggerRate(0)
	,m_start_current(-1)
	,m_stop_current(-1)
	,m_avg_current(-1)
	,m_max_current(-1)
	,m_min_current(-1)
	,m_run_intervall_micros(0){}

RunSummary * RunSummary::Fake_RunSummary(){
  RunSummary * sum = new RunSummary();
  return sum; 
}

void RunSummary::printAllValues() const {
	std::cout << "run number: " << m_run << std::endl;
	std::cout << "run sequence name: " << m_sequenceName << std::endl;
	std::cout << "global configuration key: " << m_globalConfKey << std::endl;
	std::cout << "run start time (as timestamp): " << m_start_time_ll << std::endl;
	std::cout << "run start time (as date): " << m_start_time_str << std::endl;
	std::cout << "run stop time (as timestamp): " << m_stop_time_ll << std::endl;
	std::cout << "run stop time (as date): " << m_stop_time_str << std::endl;
	std::cout << "fill number: " << m_fill << std::endl;
	std::cout << "beam energy (GeV): " << m_energy << std::endl;
	std::cout << "lumisections in the run: " << m_lumisections << std::endl;
	std::cout << "HLT key: " << m_HLTKey << std::endl;
	std::cout << "HLT key description: " << m_HLTKeyDesc << std::endl;
	std::cout << "Event number: " << m_eventNumber << std::endl;
	std::cout << "Trigger number: " << m_triggerNumber << std::endl;
	std::cout << "Average trigger rate: " << m_avgTriggerRate << std::endl;
	std::cout << "initial current " << m_start_current << std::endl;
	std::cout << "final current " << m_stop_current << std::endl;
	std::cout << "average current " << m_avg_current << std::endl;
	std::cout << "minimum current " << m_min_current << std::endl;
	std::cout << "maximum current " << m_max_current << std::endl;
	std::cout << "change current run time interval in nanoseconds " << m_run_intervall_micros << std::endl;
 }

void RunSummary::print(std::stringstream& ss) const {
	ss << "run number: " << m_run << std::endl
	   << "run sequence name: " << m_sequenceName << std::endl
	   << "global configuration key: " << m_globalConfKey << std::endl
	   << "run start time (as timestamp): " << m_start_time_ll << std::endl
	   << "run start time (as date): " << m_start_time_str << std::endl
	   << "run stop time (as timestamp): " << m_stop_time_ll << std::endl
	   << "run stop time (as date): " << m_stop_time_str << std::endl
	   << "fill number: " << m_fill << std::endl
	   << "beam energy (GeV): " << m_energy << std::endl
	   << "lumisections in the run: " << m_lumisections << std::endl
	   << "HLT key: " << m_HLTKey << std::endl
	   << "HLT key description: " << m_HLTKeyDesc << std::endl
	   << "Event number: " << m_eventNumber << std::endl
	   << "Trigger number: " << m_triggerNumber << std::endl
	   << "Average trigger rate: " << m_avgTriggerRate << std::endl
	   << "initial current " << m_start_current << std::endl
	   << "final current " << m_stop_current << std::endl
	   << "average current " << m_avg_current << std::endl
	   << "minimum current " << m_min_current << std::endl
	   << "maximum current " << m_max_current << std::endl
	   << "change current run time interval in nanoseconds " << m_run_intervall_micros << std::endl;
}

std::ostream& operator<< (std::ostream& os, RunSummary runSummary) {
	std::stringstream ss;
	runSummary.print(ss);
	os << ss.str();
	return os;
}
/*
std::vector<std::string> RunSummary::getSubdtIn() const{
  std::vector<std::string> v; 
  for (size_t i =0; i<m_subdt_in.size(); i++){
    if (m_subdt_in[i]==0) {
      v.push_back("PIXEL");
    }
    if (m_subdt_in[i]==1) {
      v.push_back("TRACKER");
    }
    if (m_subdt_in[i]==2) {
      v.push_back("ECAL");
    }
    if (m_subdt_in[i]==3) {
      v.push_back("HCAL");
    }
    
    if (m_subdt_in[i]==4) {
      v.push_back("DT");
    }  
    if (m_subdt_in[i]==5) {
      v.push_back("CSC");
    }   
    if (m_subdt_in[i]==6) {
      v.push_back("RPC");
    }  
  }
  return v;
}
*/
