#ifndef _xdaqCollector_h_
#define _xdaqCollector_h_

#include <pthread.h>

#include "toolbox/Task.h"
#include "xdata/String.h"
#include "xdata/Integer.h"
#include "xdata/Boolean.h"
#include "xdaq/Application.h"
#include "xgi/Utils.h"
#include "xgi/Method.h"

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTTPHTMLHeader.h"
#include "cgicc/HTMLClasses.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementRootT.h"
#include "DQMServices/Core/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/Core/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/CollectorRoot.h"

#include "DQMServices/XdaqCollector/interface/StateMachine.h"
#include <iostream>

class XdaqCollector : public dqm::StateMachine, public xdata::ActionListener
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
  class DummyConsumerServer : public CollectorRoot, public toolbox::Task
    {
      
    public:
      virtual void process(){}
      void enableClients(){enableClients_ = true;}
      void disableClients(){enableClients_ = false;}
      DummyConsumerServer(int port) : CollectorRoot("EvF",port), 
	toolbox::Task("Collector")
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
	  std::cout << "DCS instance at " << std::hex << (int) dcs << std::dec 
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
      xdata::Boolean enableClients_;

    };
  
  void actionPerformed (xdata::Event& e);  
  xdata::Integer port_;
  xdata::Boolean enableClients_;
};

#endif
