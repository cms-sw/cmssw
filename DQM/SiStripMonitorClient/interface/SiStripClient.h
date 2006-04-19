#ifndef _SiStripClient_h_
#define _SiStripClient_h_


#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Components/interface/UpdateObserver.h"

#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"

#include <vector>
#include <string>
#include <iostream>


class SiStripClient : public DQMBaseClient, 
			 public dqm::UpdateObserver
{
public:
  
  // You always need to have this line! Do not remove:
  XDAQ_INSTANTIATOR();

  // The class constructor:  
  SiStripClient(xdaq::ApplicationStub *s);

  // implement the method that outputs the page with the widgets (declared in DQMBaseClient):
  void general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  // the method which answers all HTTP requests of the form ".../Request?RequestID=..."
  void handleWebRequest(xgi::Input * in, xgi::Output * out);

  // this obligatory method is called whenever the client enters the "Configured" state:
  void configure();

  // this obligatory method is called whenever the client enters the "Enabled" state:
  void newRun();

  // this obligatory method is called whenever the client enters the "Halted" state:
  void endRun();

  // this obligatory method is called by the Updater component, whenever there is an update 
  void onUpdate() const;


public:

  // this client has a web interface:  
  SiStripWebInterface * webInterface_p;
};

// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(SiStripClient)

#endif
