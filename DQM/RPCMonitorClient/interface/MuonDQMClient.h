#ifndef MuonDQMClient_H
#define MuonDQMClient_H

/** \class MuonDQMClient
 * *
 *  State machine DQM Client for Muons. Owns a web interface.
 *
 *  $Date: 2006/04/05 15:45:08 $
 *  $Revision: 1.4 $
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

class QTestConfigurationParser;
class QTestConfigure;
class QTestEnabler;
class QTestStatusChecker;

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


public:

  /// MuonDQMClient has a web interface:  
  MuonWebInterface * webInterface_p;

  QTestConfigurationParser * qtParser;
  QTestConfigure * qtConfigurer;
  QTestEnabler * qtEnabler;
  QTestStatusChecker * qtChecker;

};

// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(MuonDQMClient)

#endif
