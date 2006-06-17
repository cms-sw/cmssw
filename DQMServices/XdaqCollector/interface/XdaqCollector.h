#ifndef _xdaqCollector_h_
#define _xdaqCollector_h_

#include <pthread.h>

#include "toolbox/include/Task.h"
#include "xdata/include/xdata/String.h"
#include "xdata/include/xdata/Integer.h"
#include "xdaq/include/xdaq/Application.h"
#include "xgi/include/xgi/Utils.h"
#include "xgi/include/xgi/Method.h"

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTTPHTMLHeader.h"
#include "cgicc/HTMLClasses.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"
#include "DQMServices/CoreROOT/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/UI/interface/CollectorRoot.h"

#include "DQMServices/Components/interface/StateMachine.h"
#include <iostream>

class XdaqCollector : public dqm::StateMachine
{
	
 public:
  
  XDAQ_INSTANTIATOR();
  XdaqCollector(xdaq::ApplicationStub * s);

  void Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  void general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  void css(xgi::Input  *in,
	   xgi::Output *out) throw (xgi::exception::Exception);

  
protected:

  void configureAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);
  
  void enableAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);
    
  void suspendAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);

  void resumeAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);

  void haltAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);

  void nullAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception);

 private:
  class DummyConsumerServer : public CollectorRoot, public Task
    {
      
    public:
      virtual void process(){}
      DummyConsumerServer(int port) : CollectorRoot("EvF",port), Task("Collector")
	{
	  inputAvail_=true;
	}
      int svc(){run(); return 0;}
      static CollectorRoot *instance(int port){
	if(instance_==0)
	  instance_ = new DummyConsumerServer(port);
	return instance_;
      }
      static void start()
	{
	  std::cout << "calling activate " << std::endl;
	  ((DummyConsumerServer*)instance_)->activate();
	}
      static void stopAndKill()
	{
	  DummyConsumerServer *dcs = (DummyConsumerServer*)instance_;
	  std::cout << "DCS instance at " << hex << (int) dcs << dec 
		    << std::endl;
	  std::cout << "Attempting to kill thread " << std::endl;
	  dcs->kill();
	  std::cout << "killed, deleting instance and resetting " << std::endl;
	  delete dcs;
	  instance_ = 0;
	}

      static CollectorRoot *instance(){
	return instance_;
      }
    private:
      static CollectorRoot * instance_;
    };
  
  
  xdata::Integer port_;
};

#endif
