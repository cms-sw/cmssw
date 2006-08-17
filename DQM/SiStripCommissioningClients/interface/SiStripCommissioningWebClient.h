#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQM/SiStripCommon/interface/SiStripEnumeratedTypes.h"
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
  
 private: // ----- private methods -----
  
  void scheduleCustomRequest( std::multimap<std::string,std::string> ) throw ( xgi::exception::Exception );
  
  void defineWidgets();
  
 private: // ----- private data members -----
  
  /** */
  SiStripCommissioningClient* client_;
  /** */
  MonitorUserInterface* mui_;
  
};

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H

