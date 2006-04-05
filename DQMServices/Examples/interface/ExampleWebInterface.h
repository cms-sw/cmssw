#ifndef _DQMServices_Examples_ExampleWebInterface_h_
#define _DQMServices_Examples_ExampleWebInterface_h_

/*
  This class is an example web interface that can be instantiated in a DQM client. 
  Such web interfaces inherit the ability to react to widget requests from the 
  WebInterface class of WebComponents.
*/

#include "DQMServices/WebComponents/interface/WebInterface.h"

class ExampleWebInterface : public WebInterface
{

public:

  ExampleWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);

  /*
    you need to implement this function if you have widgets that invoke custom-made
    methods defined in your client
  */
  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  /*
    An example custom-made method that we want to bind to a widget
  */
  void CustomRequestResponse(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

private:

};

#endif
