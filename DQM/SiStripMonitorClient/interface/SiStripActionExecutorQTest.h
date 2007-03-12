// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/12/07

#ifndef DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H
#define DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H

#include <string>

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"

class MonitorUserInterface;

class SiStripActionExecutorQTest: public SiStripActionExecutor {
  public:
    SiStripActionExecutorQTest(): SiStripActionExecutor() {}
    virtual ~SiStripActionExecutorQTest() {}

    // @arguments
    //   poMui  MonitorUserInterface for which QTests are assigned
    // @return
    //   summary string
    virtual std::string 
      getQTestSummary( const MonitorUserInterface *poMUI) const;

    virtual std::string
      getQTestSummaryLite( const MonitorUserInterface *poMUI) const;
};

#endif // DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H
