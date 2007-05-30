#ifndef L1TClient_h
#define L1TClient_h

/** \class L1TClient
 * *
 *  State machine DQM Client for L1 Trigger. Owns a web interface.
 *
 *  $Date: 2007/04/23 16:04:24 $
 *  $Revision: 1.2 $
 *  \author Lorenzo Agostino
  */


#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Components/interface/UpdateObserver.h"

#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include "DQM/L1TMonitorClient/interface/TriggerWebInterface.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class SubscriptionHandle;

class QTestHandle;

class L1TClient : public DQMBaseClient, 
			       public dqm::UpdateObserver
{
public:
  
  /// You always need to have this line! Do not remove:
  XDAQ_INSTANTIATOR();

  ///Constructor:  
  L1TClient(xdaq::ApplicationStub *s);

  /// Override DQMBaseClient::general, which outputs the web interface page:
  void general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  /// Handles all HTTP requests
  void handleWebRequest(xgi::Input * in, xgi::Output * out);

  /// this obligatory method is called whenever the client enters the "Configured" state:
  void configure();

  /// this obligatory method is called whenever the client enters the "Enabled" state:
  void newRun();

  /// this obligatory method is called whenever the client enters the "Halted" state:
  void endRun();

  // this obligatory method is called by the Updater component, whenever there is an update 
  void onUpdate() const;
  
  
  void CreateDQMPage(xgi::Input * in, xgi::Output * out);
  void CreateMenuPage(xgi::Input * in, xgi::Output * out);
  void CreateStatusPage(xgi::Input * in, xgi::Output * out);
  void CreateDebugPage(xgi::Input * in, xgi::Output * out);
  void CreateDisplayPage(xgi::Input * in, xgi::Output * out);
  

private:

  void checkGolbalQTStatus() const;   
  void checkDetailedQTStatus() const; 

private:

  

  /// L1TClient has a web interface:  
  TriggerWebInterface * webInterface_p;

  SubscriptionHandle *subscriber;
  QTestHandle * qtHandler;
  
  bool qtestsConfigured;
  bool meListConfigured;
  mutable bool qtestalreadyrunning;
  mutable std::ofstream logFile;    
  mutable unsigned int QTFailed;    
  mutable unsigned int QTCritical;   
  std::string url; 

};

// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(L1TClient)

#endif
