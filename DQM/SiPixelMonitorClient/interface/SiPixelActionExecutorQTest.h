// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/12/07

#ifndef DQM_SIPIXELMONITORCLIENT_SIPIXELACTIONEXECUTORQTEST_H
#define DQM_SIPIXELMONITORCLIENT_SIPIXELACTIONEXECUTORQTEST_H

#include <string>
#include <ostream>
#include <memory>

#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelXMLTags.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class SiPixelActionExecutorQTest: public SiPixelActionExecutor {
  public:
    SiPixelActionExecutorQTest(); 
    virtual ~SiPixelActionExecutorQTest() {}

    // @arguments
    //   poMui  MonitorUserInterface for which QTests are assigned
    // @return
    //   summary string
    virtual std::string 
      //getQTestSummary( const MonitorUserInterface *poMUI);
      getQTestSummary( const DaqMonitorBEInterface *poBEI);

    virtual std::string
      //getQTestSummaryLite( const MonitorUserInterface *poMUI);
      getQTestSummaryLite( const DaqMonitorBEInterface *poBEI);

    virtual std::string
      //getQTestSummaryXML( const MonitorUserInterface *poMUI);
      getQTestSummaryXML( const DaqMonitorBEInterface *poBEI);

    virtual std::string
      //getQTestSummaryXMLLite( const MonitorUserInterface *poMUI);
      getQTestSummaryXMLLite( const DaqMonitorBEInterface *poBEI);

  private:
    std::ostream &getQTestSummary_( std::ostream		&roOut,
				    //const MonitorUserInterface  *poMUI,
				    const DaqMonitorBEInterface  *poBEI,
				    const dqm::XMLTag::TAG_MODE &reMODE);
    //void createQTestSummary_( const MonitorUserInterface *poMUI);
    void createQTestSummary_( const DaqMonitorBEInterface *poBEI);

    bool			       bSummaryTagsNotRead_;
    std::auto_ptr<dqm::XMLTagWarnings> poXMLTagWarnings_;
    std::auto_ptr<dqm::XMLTagErrors>   poXMLTagErrors_;
};

#endif // DQM_SIPIXELMONITORCLIENT_SIPIXELACTIONEXECUTORQTEST_H
