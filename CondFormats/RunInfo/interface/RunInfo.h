#ifndef RunInfo_h
#define RunInfo_h

#include <iostream>
#include<vector>

/*
 *  \class RunInfo
 *  
 *  hosting run information, above all the run start and stop time, the list of fed joining, the .  
 *
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (Oct-10-2008)
 *
*/

class RunInfo {
public:
  
  int m_run;
  long long m_start_time_ll;
  std::string m_start_time_str;
  long long m_stop_time_ll;
  std::string m_stop_time_str;
  std::vector<int> m_fed_in;
  float m_start_current;
  float m_stop_current;
  float m_avg_current;
  float m_max_current;
  float m_min_current;
  float m_run_intervall_micros;
  std::vector<float> m_current;
  std::vector<float> m_times_of_currents;
    
  RunInfo();
  virtual ~RunInfo(){};
  static RunInfo* Fake_RunInfo();
    
  void printAllValues() const;

  
};


#endif
