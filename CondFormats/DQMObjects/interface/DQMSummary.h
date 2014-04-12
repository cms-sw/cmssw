#ifndef CondFormats_DQMObjects_DQMSummary_h
#define CondFormats_DQMObjects_DQMSummary_h

#include <iostream>
#include <string>
#include <vector>

/*
 *  \class DQMSummary
 *  
 *  hosting DQM information  
 *
 *  \author Salvatore Di Guida (diguida) - CERN (Mar-27-2009)
 *
*/

class DQMSummary {
 public:
  long long m_run;
  struct RunItem {
    long long m_lumisec;
    struct LumiItem {
      std::string m_subsystem;
      std::string m_reportcontent;
      std::string m_type;
      double m_status;
    };
    std::vector<LumiItem> m_lumisummary;
  };
  DQMSummary(){}
  virtual ~DQMSummary(){}
  std::vector<RunItem> m_summary;
  void printAllValues() const {
    std::cout << "run number = " << m_run << std::endl;
    std::vector<RunItem>::const_iterator runIt;
    for(runIt = m_summary.begin(); runIt != m_summary.end(); ++runIt) {
      std::cout << "--- lumisection = " << runIt->m_lumisec << std::endl;
      std::vector<RunItem::LumiItem>::const_iterator lumiIt;
      for(lumiIt = runIt->m_lumisummary.begin(); lumiIt != runIt->m_lumisummary.end(); ++lumiIt) {
	std::cout << "------ subsystem: " << lumiIt->m_subsystem 
		  << ", report content: " << lumiIt->m_reportcontent
		  << ", type: " << lumiIt->m_type
		  << ", status = " << lumiIt->m_status
		  << std::endl;
      }
    }
  }
};

#endif
