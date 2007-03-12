// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/12/07

#ifndef DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H
#define DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H

#include <string>

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"

namespace edm {
  class ParameterSet;
}
class MonitorUserInterface;

class SiStripActionExecutorQTest: public SiStripActionExecutor {
  public:
    SiStripActionExecutorQTest( const edm::ParameterSet &roPARAMETER_SET); 
    virtual ~SiStripActionExecutorQTest() {}

    // @arguments
    //   poMui  MonitorUserInterface for which QTests are assigned
    // @return
    //   summary string
    virtual std::string 
      getQTestSummary( const MonitorUserInterface *poMUI) const;

    virtual std::string
      getQTestSummaryLite( const MonitorUserInterface *poMUI) const;

  private:
    const std::string oQTEST_CONFIG_FILE_;
};

#endif // DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H
