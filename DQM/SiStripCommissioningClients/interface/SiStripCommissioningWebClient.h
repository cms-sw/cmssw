#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <string>
#include <map>

class SiStripCommissioningClient;

class SiStripCommissioningWebClient : public WebInterface {

 public:
  
  SiStripCommissioningWebClient( SiStripCommissioningClient*,
				 std::string, 
				 std::string, 
				 MonitorUserInterface** mui );
  ~SiStripCommissioningWebClient() {;}
  
  virtual void handleCustomRequest( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception );
  
 private:
  
  void defineWidgets();
  
  /** */
  SiStripCommissioningClient* client_;
  /** */
  MonitorUserInterface* mui_;
  
};

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H

