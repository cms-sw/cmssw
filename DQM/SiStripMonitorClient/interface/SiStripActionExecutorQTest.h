// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/12/07

#ifndef DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H
#define DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H

#include <string>
#include <ostream>
#include <memory>

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripXMLTags.h"

class DaqMonitorBEInterface;

class SiStripActionExecutorQTest: public SiStripActionExecutor {
  public:
    SiStripActionExecutorQTest(); 
    virtual ~SiStripActionExecutorQTest() {}

    // @arguments
    //   poMui  DaqMonitorBEInterface for which QTests are assigned
    // @return
    //   summary string
    virtual std::string 
      getQTestSummary( const DaqMonitorBEInterface *poBEI);

    virtual std::string
      getQTestSummaryLite( const DaqMonitorBEInterface *poBEI);

    virtual std::string
      getQTestSummaryXML( const DaqMonitorBEInterface *poBEI);

    virtual std::string
      getQTestSummaryXMLLite( const DaqMonitorBEInterface *poBEI);

  private:
    std::ostream &getQTestSummary_( std::ostream                &roOut,
                                    const DaqMonitorBEInterface  *poBEI,
                                    const dqm::XMLTag::TAG_MODE &reMODE);
    void createQTestSummary_( const DaqMonitorBEInterface *poBEI);

    bool                               bSummaryTagsNotRead_;
    std::auto_ptr<dqm::XMLTagWarnings> poXMLTagWarnings_;
    std::auto_ptr<dqm::XMLTagErrors>   poXMLTagErrors_;
};

#endif // DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H
