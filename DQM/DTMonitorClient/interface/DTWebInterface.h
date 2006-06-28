#ifndef DTWebInterface_H
#define DTWebInterface_H

/** \class  DTWebInterface
 *  Class that creates the web interface to control the DTDQMClient
 *  
 *  $Date: 2006/04/24 09:57:38 $
 *  $Revision: 1.1 $
 *  \author Marco Zanetti from Ilaria Segoni example
 */


#include "DQMServices/WebComponents/interface/WebInterface.h"

class DTWebInterface : public WebInterface {

public:

  DTWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);

  /*
    you need to implement this function if you have widgets that invoke custom-made
    methods defined in your client
  */
  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  /*
    An example custom-made method that we want to bind to a widget
  */
  void checkQTGlobalStatus(xgi::Input * in, xgi::Output * out , bool start) throw (xgi::exception::Exception);
  void checkQTDetailedStatus(xgi::Input * in, xgi::Output * out , bool start) throw (xgi::exception::Exception);
  void checkNoiseStatus(xgi::Input * in, xgi::Output * out , bool start) throw (xgi::exception::Exception);

  bool globalQTStatusRequest(){return performCheckingQTGlobalStatus;}
  bool detailedQTStatusRequest(){return performCheckingQTDetailedStatus;}
  bool noiseStatus(){return performCheckingNoiseStatus;}

private:

  bool performCheckingQTGlobalStatus;
  bool performCheckingQTDetailedStatus;

  bool performCheckingNoiseStatus;
};

#endif
