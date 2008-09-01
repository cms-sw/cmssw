#ifndef RunNumber_h
#define RunNumber_h

#include <iostream>
#include<vector>

/*
 *  \class RunNumber
 *  
 *  hosting runinfo information, above all the runnumber 
 *
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (June-12-2008)
 *
*/

namespace runinfo_test{
class RunNumber {
public:
  struct Item {
    Item(){}
    ~Item(){}
    int m_run;
    long long  m_id_start;
    long long  m_id_stop;
    std::string m_number;
    std::string m_name;
    signed long long m_start_time_sll;
    std::string m_start_time_str;
    signed long long m_stop_time_sll;
    std::string m_stop_time_str;
    int  m_lumisections;
    std::vector<std::string> m_subdt_joined;
    std::vector<int> m_subdt_in;
    enum subdet { PIXEL, TRACKER, ECAL, HCAL, DT, CSC,RPC };  
  };

 
 
  RunNumber();
  virtual ~RunNumber(){}
  typedef std::vector<Item>::const_iterator ItemIterator;
  std::vector<Item>  m_runnumber;
};

}
#endif
