#include "DQMServices/Core/interface/MonitorData.h"
#include "DQMServices/Core/interface/MonitorDaemon.h"

#include "SealKernel/Exception.h"
#include "SealBase/SharedLibrary.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>

using namespace dqm::monitor_data;

using std::cout; using std::endl; using std::cerr;
using std::string;

RootMonitorThread * MonitorDaemon::od = 0;

MonitorDaemon::MonitorDaemon(const edm::ParameterSet &pset)
{
  destination_address =
    pset.getUntrackedParameter<string>("DestinationAddress","localhost"); 
  send_port = pset.getUntrackedParameter<int>("SendPort", COLLECTOR_PORT);
  primary_delay = pset.getUntrackedParameter<int>("UpdateDelay", SEND_PERIOD);
  name_as_source = 
    pset.getUntrackedParameter<string>("NameAsSource", SOURCE_NAME);
  auto_instantiating = pset.getUntrackedParameter<bool>("AutoInstantiate", 
							false);

  reconnect_delay = 
    pset.getUntrackedParameter<int>("reconnect_delay", RECONNECT_DELAY);
  maxAttempts2Reconnect = 
    pset.getUntrackedParameter<int>("maxAttempts2Reconnect", MAX_RECON);
  cout << " MonitorDaemon constructor called, auto = " 
	    << auto_instantiating << endl; 
  
  // get hold of back-end interface instance
  DaqMonitorBEInterface * dbe = 0;
  try
    {
      dbe = edm::Service<DaqMonitorBEInterface>().operator->();
    }
  catch(edm::Exception e1)
    {
      cout << e1.what() << endl;
      exit(-1);
    }
  catch(std::exception& e)
    {
      cerr << " std::Exception:\n" << e.what() << endl;
      exit(-2);
    }
  catch(seal::SharedLibraryError& e)
    {
      cerr << " sharedliberror\n" << e.explainSelf() << endl;
      exit(-3);
    }
  catch(seal::Error& e)
    {
      cerr << " seal::Error:\n" << e.explain() << endl;
      exit(-4);
    }
  catch(...)
    {
      cerr << " weird exception" << endl;
      throw;
    }
  
  if(auto_instantiating)
    rmt(destination_address, send_port, primary_delay, name_as_source, reconnect_delay);

}

MonitorDaemon::~MonitorDaemon(void)
{
  if(od)
    {
      if(od->isConnected())od->terminate();
      delete od; od = 0;
    }
}

RootMonitorThread * MonitorDaemon::rmt(string add, int p, int del, string nam, int rdel)
{
  cout << " Attempting to start a MonitorDaemon at address " << add 
       << " port " << p << endl;
  if(!od)
    {
      edm::ServiceToken tok =  
	edm::ServiceRegistry::instance().presentToken();
      // convert time to microsecs
      od = new RootMonitorThread(add, p, del*1000, nam, tok, 
				 rdel);
      setMaxAttempts2Reconnect(maxAttempts2Reconnect);
      od->release();
    }

  return od;
}

    
