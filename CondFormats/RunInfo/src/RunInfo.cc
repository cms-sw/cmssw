#include "CondFormats/RunInfo/interface/RunInfo.h"
RunInfo::RunInfo(){}

RunInfo * RunInfo::Fake_RunInfo(){
  RunInfo * sum = new RunInfo();
  sum->m_run=-1;
  sum->m_start_time_ll=-1;
  sum->m_start_time_str="null";
  sum->m_stop_time_ll=-1;
  sum->m_stop_time_str="null";
  sum->m_start_current=-1; 
  sum->m_stop_current=-1;
  sum->m_avg_current=-1;
  sum->m_max_current=-1;
  sum->m_min_current=-1;
  sum->m_run_intervall_micros=0;
  return sum; 
}



void RunInfo::printAllValues() const {
  std::cout << "run number: " << m_run << std::endl;
  std::cout << "run start time (as timestamp): " << m_start_time_ll << std::endl;
  std::cout << "run start time (as date): " << m_start_time_str << std::endl;
  std::cout << "run stop time (as timestamp): " << m_stop_time_ll << std::endl;
  std::cout << "run stop time (as date): " << m_stop_time_str << std::endl;
  std::cout << "initial current " << m_start_current << std::endl;
  std::cout << "final current " << m_stop_current << std::endl;
  std::cout << "average current " << m_avg_current << std::endl;
  std::cout << "minimum current " << m_min_current << std::endl;
  std::cout << "maximum current " << m_max_current << std::endl;
  std::cout << "run time in microseconds " << m_run_intervall_micros << std::endl;
  std::cout << "ids of fed in run: " << std::endl;
  for(size_t i = 0; i < m_fed_in.size(); i++) {
    std::cout << "---> " << m_fed_in[i] << std::endl;
  }
  std::cout << "B current in run: " << std::endl;
  for(size_t i = 0; i < m_current.size(); i++) {
    std::cout << "---> " << m_current[i] << std::endl;
  }
  std::cout << "correspondent time (from run start) in microseconds for B currents in run: " << std::endl;
  for(size_t i = 0; i < m_times_of_currents.size(); i++) {
    std::cout << "---> " << m_times_of_currents[i] << std::endl;
  }
}


