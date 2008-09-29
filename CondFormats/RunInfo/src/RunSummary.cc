#include "CondFormats/RunInfo/interface/RunSummary.h"
RunSummary::RunSummary(){
  summary.reserve(600000);
}


void RunSummary::printAllValues() const{
  for (SummaryIterator it=summary.begin(); it!=summary.end(); ++it){
  std::cout<<it->m_run << std::endl;
  std::cout<<it->m_name << std::endl;
  std::cout<<it->m_start_time_ll<<std::endl;
  std::cout<<it->m_start_time_str<<std::endl;
  std::cout<<it->m_stop_time_ll<< std::endl;
  std::cout<<it->m_stop_time_str<<std::endl;
  std::cout<<it->m_lumisections<<std::endl;
  std::cout<<it->m_hltkey<<std::endl;
  std::cout<<it->m_nevents<<std::endl;
  std::cout<<it->m_rate<<std::endl;
  for (size_t i =0; i<it->m_subdt_joining.size(); i++){
    std::cout<<it->m_subdt_joining[i]<<std::endl;
  }
for (size_t i =0; i<it->m_subdt_in.size(); i++){
    std::cout<<it->m_subdt_in[i]<<std::endl;
  }
  }
}
RunSummary::Summary RunSummary::fake_Run(){
  RunSummary::Summary s;
  s.m_run=-1;
  s.m_name="null";
  s.m_start_time_ll=-1;
  s.m_start_time_str="null";
  s.m_stop_time_ll=-1;
  s.m_stop_time_str="null";
  s.m_lumisections=-1;
  s.m_hltkey="null";
  s.m_nevents=-1;
  s.m_rate=-1;
  return s;
}
