#ifndef DTDQMClient_H
#define DTDQMClient_H

/** \class DTDQMClient
 * *
 *  State machine DQM Client for DTs. Owns a web interface.
 *
 *  $Date: 2006/05/08 12:26:41 $
 *  $Revision: 1.3 $
 *  \author Marco Zanetti from Ilaria Segoni example
 *   
  */


#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Components/interface/UpdateObserver.h"

#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include "DQM/DTMonitorClient/interface/DTWebInterface.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class SubscriptionHandle;
class QTestHandle;
class DTNoiseClient;

class DTDQMClient : public DQMBaseClient, 
			       public dqm::UpdateObserver
{
public:
  
  /// You always need to have this line! Do not remove:
  XDAQ_INSTANTIATOR();

  ///Constructor:  
  DTDQMClient(xdaq::ApplicationStub *s);

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

  /// DTDQMClient has a web interface:  
  DTWebInterface * webInterface_p;

  SubscriptionHandle *subscriber;
  QTestHandle * qtHandler;

  DTNoiseClient * noiseClient;
  
  bool qtestsConfigured;
  bool meListConfigured;
 
  mutable std::ofstream logFile;    
  mutable unsigned int QTFailed;    
  mutable unsigned int QTCritical;    

};

// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(DTDQMClient)

#endif
