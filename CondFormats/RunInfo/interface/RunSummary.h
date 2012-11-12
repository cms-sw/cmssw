#ifndef RunSummary_h
#define RunSummary_h

#include <iostream>
#include<vector>

/*
 *  \class RunSummary
 *  
 *  hosting light run information, above all the run start and stop time, the list of subdetector joining, tle number of lumisections.  
 *
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (Sep-24-2008)
 *
*/

class RunSummary {
public:
  
  int m_run;
  std::string m_name;
  long long m_start_time_ll;
  std::string m_start_time_str;
  long long m_stop_time_ll;
  std::string m_stop_time_str;
  int  m_lumisections;
  std::vector<int> m_subdt_in;
  std::string m_hltkey;
  long long m_nevents;
  float m_rate;
    
  enum subdet { PIXEL, TRACKER, ECAL, HCAL, DT, CSC,RPC };  
  
  RunSummary();
  virtual ~RunSummary(){};
  static RunSummary* Fake_RunSummary();
    
  void printAllValues() const;
  std::vector<std::string> getSubdtIn() const;
  
};


#endif
