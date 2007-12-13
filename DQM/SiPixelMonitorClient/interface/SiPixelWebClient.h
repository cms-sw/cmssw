#ifndef _DQM_SiPixelMonitorClient_SiPixelWebClient_h_
#define _DQM_SiPixelMonitorClient_SiPixelWebClient_h_

#include "DQMServices/XdaqCollector/interface/DQMBaseClient.h"
#include "DQMServices/XdaqCollector/interface/Updater.h"
#include "DQMServices/XdaqCollector/interface/UpdateObserver.h"

#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include "DQM/SiPixelMonitorClient/interface/SiPixelWebInterface.h"

#include <vector>
#include <string>
#include <iostream>


class SiPixelWebClient : public DQMBaseClient, 
			 public dqm::UpdateObserver
{
public:
  
  // You always need to have this line! Do not remove:
  XDAQ_INSTANTIATOR();

  // The class constructor:  
  SiPixelWebClient(xdaq::ApplicationStub *s);

  // implement the method that outputs the page with the widgets (declared in DQMBaseClient):
  void general(xgi::Input * in, 
               xgi::Output * out ) 
	       throw (xgi::exception::Exception);

  // the method which answers all HTTP requests of the form ".../Request?RequestID=..."
  void handleWebRequest(xgi::Input * in, 
                        xgi::Output * out);

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
  SiPixelWebInterface * webInterface_p;
  // and an ActionExecutor:
  SiPixelActionExecutor * actionExecutor_;

private:

  void checkCustomRequests() const;
  void setupQTest() const;

  int updateFrequencyForTrackerMap_;
  int updateFrequencyForBarrelSummary_;
  int updateFrequencyForEndcapSummary_;
  int updateFrequencyForGrandBarrelSummary_;
  int updateFrequencyForGrandEndcapSummary_;
  int messageLimitForQTests_;
  int source_type_;
};

// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(SiPixelWebClient)

#endif
