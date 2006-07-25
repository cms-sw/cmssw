#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H

#include "DQMServices/WebComponents/interface/WebInterface.h"

class SiStripCommissioningClient;

class SiStripCommissioningWebClient : public WebInterface {

 public:
  
  SiStripCommissioningWebClient( SiStripCommissioningClient*,
				 std::string, 
				 std::string, 
				 MonitorUserInterface** mui );
  ~SiStripCommissioningWebClient();
  
  virtual void handleCustomRequest( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception );
  
 private: // ----- private methods -----
  
  void createSummaryHistos( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception );  
  void createTrackerMap( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception ); 
  void uploadToConfigDb( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception ); 

 private: // ----- private data members -----
  
  /** */
  SiStripCommissioningClient* client_;
  /** */
  MonitorUserInterface* mui_;

};

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H

