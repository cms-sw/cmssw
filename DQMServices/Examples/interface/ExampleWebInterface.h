#ifndef _DQMServices_Examples_ExampleWebInterface_h_
#define _DQMServices_Examples_ExampleWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"

class ExampleWebInterface : public WebInterface
{

public:

  ExampleWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);

  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  void CustomRequestResponse(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

private:

};

#endif
