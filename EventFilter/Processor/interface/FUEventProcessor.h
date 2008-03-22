#ifndef FUEVENTPROCESSOR_H
#define FUEVENTPROCESSOR_H 1


#include "EventFilter/Utilities/interface/StateMachine.h"
#include "EventFilter/Utilities/interface/RunBase.h"
#include "EventFilter/Utilities/interface/Css.h"
#include "EventFilter/Utilities/interface/Exception.h"


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
    void getTriggerReport(toolbox::Event::Reference e)
      throw (toolbox::fsm::exception::Exception);

    // trigger prescale callbacks
    xoap::MessageReference getPsReport(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    xoap::MessageReference getLsReport(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    xoap::MessageReference putPrescaler(xoap::MessageReference msg)
      throw (xoap::exception::Exception);

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
    std::string triggerReportToString(const edm::TriggerReport& tr);
    void        printTriggerReport(const edm::TriggerReport& tr);

    // HyperDAQ related functions
    void defaultWebPage(xgi::Input *in,xgi::Output *out)
      throw(xgi::exception::Exception);
    void taskWebPage(xgi::Input *,xgi::Output *,const std::string &);
    void spotlightWebPage(xgi::Input *,xgi::Output *)
      throw(xgi::exception::Exception);
    void moduleWeb(xgi::Input *in,xgi::Output *out) throw(xgi::exception::Exception);
    void modulePs(xgi::Input  *in, xgi::Output *out)throw(xgi::exception::Exception);
    void microState(xgi::Input *in,xgi::Output *out)throw(xgi::exception::Exception);
    void jsGen(xgi::Input *in,xgi::Output *out,std::string url) 
      throw (xgi::exception::Exception);
    void css(xgi::Input *in,xgi::Output *out) throw (xgi::exception::Exception)
    {
      css_.css(in,out);
    }
    void attachDqmToShm()   throw (evf::Exception);
    void detachDqmFromShm() throw (evf::Exception);

    // calculate monitoring information in separate thread
    void startMonitoringWorkLoop() throw (evf::Exception);
    bool monitoring(toolbox::task::WorkLoop* wl);

    
  private:
    //
    // member data
    //
    
    // finite state machine
    evf::StateMachine               fsm_;
    
    // event processor
    edm::EventProcessor             *evtProcessor_;
    edm::ServiceToken                serviceToken_;    
    bool                             servicesDone_;

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
    xdata::Boolean                   isRunNumberSetter_;
    bool                             outprev_;
    std::vector<edm::ModuleDescription const*> descs_; //module description array
    std::map<std::string,int>        modmap_;
    
    // dqm monitor thread configuration
    xdata::String                    dqmCollectorAddr_;
    xdata::Integer                   dqmCollectorPort_;
    xdata::Integer                   dqmCollectorDelay_;
    xdata::Integer                   dqmCollectorReconDelay_;
    xdata::String                    dqmCollectorSourceName_;

    // xdaq parameters relevant to trigger-report / prescales
    xdata::String                    triggerReportAsString_;
    xdata::String                    prescalerAsString_;

    // xdaq monitoring
    xdata::UnsignedInteger32         monSleepSec_;
    struct timeval                   monStartTime_;

    // workloop / action signature for monitoring
    toolbox::task::WorkLoop         *wlMonitoring_;      
    toolbox::task::ActionSignature  *asMonitoring_;

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

    
    // HyperDAQ related
    Css                              css_;

    // Misc
    std::string                      reasonForFailedState_;
    
  };
  
} // namespace evf


#endif
