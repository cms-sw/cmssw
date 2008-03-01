// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/12/07

#ifndef DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H
#define DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H

#include <string>
#include <ostream>
#include <memory>

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripXMLTags.h"

class DQMStore;

class SiStripActionExecutorQTest: public SiStripActionExecutor {
  public:
    SiStripActionExecutorQTest(); 
    virtual ~SiStripActionExecutorQTest() {}

    // @arguments
    //   poMui  DQMStore for which QTests are assigned
    // @return
    //   summary string
    virtual std::string 
      getQTestSummary( const DQMStore *poDQMStore);

    virtual std::string
      getQTestSummaryLite( const DQMStore *poDQMStore);

    virtual std::string
      getQTestSummaryXML( const DQMStore *poDQMStore);

    virtual std::string
      getQTestSummaryXMLLite( const DQMStore *poDQMStore);

  private:
    std::ostream &getQTestSummary_( std::ostream                &roOut,
                                    const DQMStore  *poDQMStore,
                                    const dqm::XMLTag::TAG_MODE &reMODE);
    void createQTestSummary_( const DQMStore *poDQMStore);

    bool                               bSummaryTagsNotRead_;
    std::auto_ptr<dqm::XMLTagWarnings> poXMLTagWarnings_;
    std::auto_ptr<dqm::XMLTagErrors>   poXMLTagErrors_;
};

#endif // DQM_SISTRIPMONITORCLIENT_SISTRIPACTIONEXECUTORQTEST_H
