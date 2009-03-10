#ifndef FUEVENTPROCESSOR_H
#define FUEVENTPROCESSOR_H 1

#include "EventFilter/Processor/src/TriggerReportHelpers.h"

#include "EventFilter/Utilities/interface/StateMachine.h"
#include "EventFilter/Utilities/interface/RunBase.h"
#include "EventFilter/Utilities/interface/Css.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/SquidNet.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"

#include "xdaq/Application.h"
#include "xdaq/NamespaceURI.h"

#include "xdata/String.h"
#include "xdata/Integer.h"
#include "xdata/Boolean.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/ActionListener.h"
#include "xdata/InfoSpaceFactory.h"

#include "xgi/Input.h"
#include "xgi/Output.h"
#include "xgi/exception/Exception.h"

#include <sys/time.h>

#include <list>
#include <vector>
#include <map>

namespace edm {
  class EventProcessor;
}

namespace evf
{
  /* to be filled in with summary from paths */
  struct filter {

  };
  
  
  class FUEventProcessor : public xdaq::Application,
			   public xdata::ActionListener
  {
  public:
    //
    // construction/destruction
    //
    XDAQ_INSTANTIATOR();
    FUEventProcessor(xdaq::ApplicationStub *s);
    virtual ~FUEventProcessor();
    

    //
    // member functions
    //

    // trigger report callback
    bool getTriggerReport(bool useLock)
      throw (toolbox::fsm::exception::Exception);

    // work loop functions to be executed during transitional states (async)
    bool configuring(toolbox::task::WorkLoop* wl);
    bool enabling(toolbox::task::WorkLoop* wl);
    bool stopping(toolbox::task::WorkLoop* wl);
    bool halting(toolbox::task::WorkLoop* wl);

    // fsm soap command callback
    xoap::MessageReference fsmCallback(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    
    // initialize the cmssw event processor
    void initEventProcessor();
    edm::EventProcessor::StatusCode stopEventProcessor();
    
    // xdata:ActionListener interface
    void actionPerformed(xdata::Event& e);
    
    // trigger report related helper functions
    //    std::string triggerReportToString(const edm::TriggerReport& tr);
    //    void triggerReportToTable(const edm::TriggerReport& tr);
    //    void        printTriggerReport(const edm::TriggerReport& tr);

    // HyperDAQ related functions
    void defaultWebPage(xgi::Input *in,xgi::Output *out)
      throw(xgi::exception::Exception);
    void taskWebPage(xgi::Input *,xgi::Output *,const std::string &);
    void spotlightWebPage(xgi::Input *,xgi::Output *)
      throw(xgi::exception::Exception);
    void moduleWeb(xgi::Input *in,xgi::Output *out) throw(xgi::exception::Exception);
    void serviceWeb(xgi::Input *in,xgi::Output *out) throw(xgi::exception::Exception);
    void modulePs(xgi::Input  *in, xgi::Output *out)throw(xgi::exception::Exception);
    void microState(xgi::Input *in,xgi::Output *out)throw(xgi::exception::Exception);
    void css(xgi::Input *in,xgi::Output *out) throw (xgi::exception::Exception)
    {
      css_.css(in,out);
    }

  private:

    void attachDqmToShm()   throw (evf::Exception);
    void detachDqmFromShm() throw (evf::Exception);

    // calculate monitoring information in separate thread
    void startMonitoringWorkLoop() throw (evf::Exception);
    bool monitoring(toolbox::task::WorkLoop* wl);

    void startScalersWorkLoop() throw (evf::Exception);
    bool scalers(toolbox::task::WorkLoop* wl);


    bool fireScalersUpdate();


    std::string logsAsString();
    void localLog(std::string);


    //
    // member data
    //
    
    // finite state machine
    evf::StateMachine               fsm_;
    
    // event processor
    edm::EventProcessor             *evtProcessor_;
    edm::ServiceToken                serviceToken_;    
    bool                             servicesDone_;
    bool                             inRecovery_;
    unsigned int                     recoveryCount_;
    bool                             triggerReportIncomplete_;

    // prescale (cmssw framework-) service
    edm::service::PrescaleService*  prescaleSvc_;
    
    // parameters published to XDAQ info space(s)
    xdata::String                    url_;
    xdata::String                    class_;
    xdata::UnsignedInteger32         instance_;
    xdata::UnsignedInteger32         runNumber_;
    xdata::Boolean                   epInitialized_; 
    xdata::String                    configString_;
    std::string                      configuration_;
    xdata::String                    sealPluginPath_;
    xdata::Boolean                   outPut_;
    xdata::UnsignedInteger32         inputPrescale_;
    xdata::UnsignedInteger32         outputPrescale_;
    xdata::UnsignedInteger32         timeoutOnStop_; // in seconds
    xdata::Boolean                   hasShMem_;
    xdata::Boolean                   hasPrescaleService_;
    xdata::Boolean                   hasModuleWebRegistry_;
    xdata::Boolean                   hasServiceWebRegistry_;
    xdata::Boolean                   isRunNumberSetter_;
    xdata::Boolean                   isPython_;
    bool                             outprev_;
    std::vector<edm::ModuleDescription const*> descs_; //module description array
    std::map<std::string,int>        modmap_;
    
    // xdaq parameters relevant to trigger-report / prescales
    //    xdata::String                    triggerReportAsString_;
    xdata::String                    prescalerAsString_;

    // xdaq monitoring
    xdata::UnsignedInteger32         monSleepSec_;
    struct timeval                   monStartTime_;

    // workloop / action signature for monitoring
    toolbox::task::WorkLoop         *wlMonitoring_;      
    toolbox::task::ActionSignature  *asMonitoring_;
    bool                             watching_;

    // workloop / action signature for scalerMonitor
    toolbox::task::WorkLoop         *wlScalers_;      
    toolbox::task::ActionSignature  *asScalers_;

    // application identifier
    std::string                      sourceId_;

    // flahslist variables
    xdata::String                    epMState_;
    xdata::String                    epmState_;
    xdata::UnsignedInteger32         nbProcessed_;
    xdata::UnsignedInteger32         nbAccepted_;
    xdata::InfoSpace                *monitorInfoSpace_;

    // flahslist variables, alt
    xdata::Integer                   epMAltState_;
    xdata::Integer                   epmAltState_;
    xdata::InfoSpace                *monitorInfoSpaceAlt_;

    // flahslist variables, legend
    xdata::String                    macro_state_legend_;
    xdata::String                    micro_state_legend_;
    xdata::InfoSpace                *monitorInfoSpaceLegend_;

    // flahslist variables, scalers
    xdata::InfoSpace                *scalersInfoSpace_;
    xdata::Table                     scalersComplete_;
    xdata::UnsignedInteger32         localLsIncludingTimeOuts_;
    xdata::UnsignedInteger32         lsTimeOut_;
    unsigned int                     firstLsTimeOut_;
    unsigned int                     residualTimeOut_;
    bool                             lastLsTimedOut_; 
    unsigned int                     lastLsWithEvents_;
    unsigned int                     lastLsWithTimeOut_;
    // flashlist variables, squids
    xdata::Boolean                   squidPresent_; 

    
    // HyperDAQ related
    Css                              css_;

    // Misc
    std::string                      reasonForFailedState_;
    fuep::TriggerReportHelpers       trh_;
    std::list<std::string>           names_;
    bool                             wlMonitoringActive_;
    bool                             wlScalersActive_;
    unsigned int                     scalersUpdateAttempted_;    
    unsigned int                     scalersUpdateCounter_;
    std::vector<std::pair<unsigned int, unsigned int> > lumiSectionsCtr_;
    std::vector<bool>                lumiSectionsTo_;
    unsigned int                     allPastLumiProcessed_;
    unsigned int                     rollingLsIndex_;
    bool                             rollingLsWrap_;
    static const unsigned int        lsRollSize_ = 20;
    SquidNet                         squidnet_;
    std::vector<std::string>         logRing_;
    unsigned int                     logRingIndex_;
    static const unsigned int        logRingSize_ = 50;
    bool                             logWrap_;
    std::string                      lsidAsString_;
    std::string                      lsidTimedOutAsString_;
    std::string                      psidAsString_;
  };
  
} // namespace evf


#endif
