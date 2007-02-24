#ifndef FUEVENTPROCESSOR_H
#define FUEVENTPROCESSOR_H 1


#include "EventFilter/Utilities/interface/RunBase.h"
#include "EventFilter/Utilities/interface/Css.h"

#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/PresenceFactory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/src/PrescaleService.h"

#include "xdaq/include/xdaq/Application.h"
#include "xdaq/NamespaceURI.h"

#include "xdaq2rc/RcmsStateNotifier.h"

#include "toolbox/fsm/FiniteStateMachine.h"
#include "toolbox/task/WorkLoopFactory.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "toolbox/task/Action.h"

#include "xdata/include/xdata/String.h"
#include "xdata/include/xdata/Integer.h"
#include "xdata/include/xdata/Boolean.h"
#include "xdata/include/xdata/UnsignedInteger32.h"
#include "xdata/ActionListener.h"

#include "xgi/include/xgi/Input.h"
#include "xgi/include/xgi/Output.h"
#include "xgi/include/xgi/exception/Exception.h"


namespace edm {
  class EventProcessor;
}

namespace evf
{
  /* to be filled in with summary from paths */
  struct filter {

  };
  
  
  class FUEventProcessor : public xdaq::Application,
			   public xdata::ActionListener,
			   public evf::RunBase
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
    
    // finite state machine command callbacks
    xoap::MessageReference fsmCallback(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    
    // finite state machine callback for entering new state
    void fsmStateChanged(toolbox::fsm::FiniteStateMachine & fsm) 
      throw (toolbox::fsm::exception::Exception);

    // trigger report callback
    void getTriggerReport(toolbox::Event::Reference e)
      throw (toolbox::fsm::exception::Exception);

    // trigger prescale callbacks
    xoap::MessageReference getPsReport(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    xoap::MessageReference setPsUpdate(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    xoap::MessageReference putPrescaler(xoap::MessageReference msg)
      throw (xoap::exception::Exception);

    // work loop functions to be executed during transitional states (async)
    bool configuring(toolbox::task::WorkLoop* wl);
    bool enabling(toolbox::task::WorkLoop* wl);
    bool stopping(toolbox::task::WorkLoop* wl);
    bool halting(toolbox::task::WorkLoop* wl);
    
    // initialize the cmssw event processor
    void initEventProcessor();
    
    // xdata:ActionListener interface
    void actionPerformed(xdata::Event& e);
    
    // trigger report related helper functions
    std::string triggerReportToString(const edm::TriggerReport& tr);
    void        printTriggerReport(const edm::TriggerReport& tr);

    // HyperDAQ related functions
    void defaultWebPage(xgi::Input *in,xgi::Output *out)
      throw(xgi::exception::Exception);
    void taskWebPage(xgi::Input *,xgi::Output *,const std::string &);
    void moduleWeb(xgi::Input *in,xgi::Output *out) throw(xgi::exception::Exception);
    void modulePs(xgi::Input  *in, xgi::Output *out)throw(xgi::exception::Exception);
    void microState(xgi::Input *in,xgi::Output *out)throw(xgi::exception::Exception);
    void jsGen(xgi::Input *in,xgi::Output *out,std::string url) 
      throw (xgi::exception::Exception);
    void css(xgi::Input *in,xgi::Output *out) throw (xgi::exception::Exception)
    {
      css_.css(in,out);
    }
    
    
  private:
    //
    // member data
    //
    
    // finite state machine
    toolbox::fsm::FiniteStateMachine fsm_;
    
    // work loops for transitional states
    toolbox::task::WorkLoop         *workLoopConfiguring_;
    toolbox::task::WorkLoop         *workLoopEnabling_;
    toolbox::task::WorkLoop         *workLoopStopping_;
    toolbox::task::WorkLoop         *workLoopHalting_;

    // action signatures for transitional states
    toolbox::task::ActionSignature  *asConfiguring_;
    toolbox::task::ActionSignature  *asEnabling_;
    toolbox::task::ActionSignature  *asStopping_;
    toolbox::task::ActionSignature  *asHalting_;

    // rcms state notifier
    xdaq2rc::RcmsStateNotifier       rcmsStateNotifier_;

    // event processor
    edm::EventProcessor             *evtProcessor_;
    edm::ServiceToken                serviceToken_;    
    bool                             servicesDone_;

    // prescale (cmssw framework-) service
    edm::service::PrescaleService*  prescaleSvc_;
    
    // parameters published to XDAQ info space(s)
    xdata::String                    stateName_;
    xdata::String                    configString_;
    xdata::String                    sealPluginPath_;
    xdata::Boolean                   outPut_;
    xdata::UnsignedInteger32         inputPrescale_;
    xdata::UnsignedInteger32         outputPrescale_;
    bool                             outprev_;
    
    // dqm monitor thread configuration
    xdata::String                    dqmCollectorAddr_;
    xdata::Integer                   dqmCollectorPort_;
    xdata::Integer                   dqmCollectorDelay_;
    xdata::Integer                   dqmCollectorReconDelay_;
    xdata::String                    dqmCollectorSourceName_;

    // xdaq parameters relevant to trigger-report / prescales
    xdata::String                    triggerReportAsString_;
    xdata::String                    prescalerAsString_;

    // HyperDAQ related
    Css                              css_;
    
  };
  
} // namespace evf


#endif
