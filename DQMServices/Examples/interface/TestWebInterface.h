#ifndef _DQMServices_WebComponents_TestWebInterface_h_
#define _DQMServices_WebComponents_TestWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"

class TestWebInterface : public WebInterface
{

 public:

  TestWebInterface(std::string _exeURL, std::string _appURL, 
		   MonitorUserInterface * _mui, dqm::Updater * _updater);

  // this function should be implemented so as to check for functions corresponding 
  // to custom requests and call them:
  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  // An example function corresponding to a custom request:
  void MyCustomRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
};

#endif
