#ifndef SiPixelHistoricInfoClient_SiPixelHistoricInfoWebInterface_h
#define SiPixelHistoricInfoClient_SiPixelHistoricInfoWebInterface_h

/* Description: this class is an example web interface that can be 
                instantiated in a DQM client. Such web interfaces 
		inherit the ability to react to widget requests from 
		the WebInterface class of WebComponents. */
		
#include "DQMServices/WebComponents/interface/WebInterface.h"


class SiPixelHistoricInfoWebInterface : public WebInterface {
public:
  SiPixelHistoricInfoWebInterface(std::string theContextURL, 
                                  std::string theApplicationURL, 
				  MonitorUserInterface** _mui_p);

  // you need to implement this function if you have widgets 
  // that invoke custom-made methods defined in your client
  void handleCustomRequest(xgi::Input* in, xgi::Output* out) throw (xgi::exception::Exception);

  // methods that we want to bind to widgets
  void saveToFile(xgi::Input* in, xgi::Output* out) throw (xgi::exception::Exception);

  // methods to get/set action flags
  bool getSaveToFile() const { return doSaveToFile; };
  void setSaveToFile(bool setsavto) { doSaveToFile = setsavto; };

private:
  // action flags that are raised in the WebInterface but executed in the Client in the onUpdate() method
  bool doSaveToFile; 
};


#endif
