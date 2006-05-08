#ifndef MuonDQMClient_H
#define MuonDQMClient_H

/** \class MuonDQMClient
 * *
 *  State machine DQM Client for Muons. Owns a web interface.
 *
 *  $Date: 2006/05/04 10:27:22 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
  */


#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Components/interface/UpdateObserver.h"

#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include "DQM/RPCMonitorClient/interface/MuonWebInterface.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class SubscriptionHandle;
class QTestHandle;

class MuonDQMClient : public DQMBaseClient, 
			       public dqm::UpdateObserver
{
public:
  
  /// You always need to have this line! Do not remove:
  XDAQ_INSTANTIATOR();

  ///Constructor:  
  MuonDQMClient(xdaq::ApplicationStub *s);

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
  

private:

  void checkGolbalQTStatus() const;   
  void checkDetailedQTStatus() const; 

private:

  

  /// MuonDQMClient has a web interface:  
  MuonWebInterface * webInterface_p;

  SubscriptionHandle *subscriber;
  QTestHandle * qtHandler;
  
  bool qtestsConfigured;
  bool meListConfigured;
 
  mutable std::ofstream logFile;    
  mutable unsigned int QTFailed;    
  mutable unsigned int QTCritical;    

};

// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(MuonDQMClient)

#endif
