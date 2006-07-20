#ifndef MuonWebInterface_H
#define MuonWebInterface_H

/** \class  MuonWebInterface
 *  Class that creates the web interface to control the MuonDQMClient
 *  
 *  $Date: 2006/04/24 09:57:38 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */


#include "DQMServices/WebComponents/interface/WebInterface.h"

class MuonWebInterface : public WebInterface
{

   public:

  	MuonWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);

  /*
    you need to implement this function if you have widgets that invoke custom-made
    methods defined in your client
  */
  	void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  /*
    An example custom-made method that we want to bind to a widget
  */
	void CheckQTGlobalStatus(xgi::Input * in, xgi::Output * out , bool start) throw (xgi::exception::Exception);
	void CheckQTDetailedStatus(xgi::Input * in, xgi::Output * out , bool start) throw (xgi::exception::Exception);

	bool globalQTStatusRequest(){return checkQTGlobalStatus;}
	bool detailedQTStatusRequest(){return checkQTDetailedStatus;}
	std::string requestType(){return request;}
private:

	bool checkQTGlobalStatus;
	bool checkQTDetailedStatus;
	std::string request;
};

#endif
