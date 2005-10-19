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

class XdaqCollector: public xdaq::Application 
{
	
 public:
  
  XDAQ_INSTANTIATOR();
  XdaqCollector(xdaq::ApplicationStub * s): xdaq::Application(s)
    {	
      clientport_ = 9090;
      sourceport_ = 9050;
      xdata::InfoSpace *sp = getApplicationInfoSpace();
      sp->fireItemAvailable("sourcePort",&sourceport_);
      sp->fireItemAvailable("clientPort",&clientport_);
      DummyConsumerServer::instance(clientport_,sourceport_);
    }
 private:
  class DummyConsumerServer : public CollectorRoot, public Task
    {
      
    public:
      virtual void process(){}
      DummyConsumerServer(int c, int s) : CollectorRoot("EvF",1,c,s), Task("Collector")
	{
	  inputAvail_=true;
	}
      int svc(){run(); return 0;}
      static CollectorRoot *instance(int cport, int sport){
	if(instance_==0)
	  instance_ = new DummyConsumerServer(cport,sport);
	((DummyConsumerServer*)instance_)->activate();
	return instance_;
      }
      
    private:
      static CollectorRoot * instance_;
    };
  
  
  xdata::Integer sourceport_;
  xdata::Integer clientport_;
};

#endif
