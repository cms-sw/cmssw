#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H

#include "DQMServices/WebComponents/interface/WebInterface.h"

class SiStripCommissioningWebClient : public WebInterface {
 public:
  
  SiStripCommissioningWebClient( std::string, 
				 std::string, 
				 MonitorUserInterface** _mui_p );
  ~SiStripCommissioningWebClient();
  
  void Default( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception );
  
  virtual void handleCustomRequest( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception );
  
 private: // ----- private methods -----
  
/*   void subscribeAll( xgi::Input* in,  */
/* 		     xgi::Output* out ) throw ( xgi::exception::Exception ); */
  
/*   int getUpdates(); */

 private: // ----- private data members -----
  
  WebPage* webpage_;
  
};

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningWebClient_H
