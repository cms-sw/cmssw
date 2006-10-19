#ifndef _DQM_SiPixelMonitorClient_SiPixelWebInterface_h_
#define _DQM_SiPixelMonitorClient_SiPixelWebInterface_h_

/*
  This class is an example web interface that can be instantiated in a DQM client. 
  Such web interfaces inherit the ability to react to widget requests from the 
  WebInterface class of WebComponents.
*/

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/Core/interface/MonitorElement.h"
class SiPixelActionExecutor;

class SiPixelWebInterface : public WebInterface
{

public:

  enum SiPixelActionType{NoAction=0, SubscribeAll=1, Summary=2, Collate=3,
                         QTestResult=4, PersistantTkMap=5, 
                         TemporaryTkMap=6, SaveData=7};

  SiPixelWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);
  ~SiPixelWebInterface();

  /*
    you need to implement this function if you have widgets that invoke custom-made
    methods defined in your client
  */
  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  void readSelectedRequest(xgi::Input * in, xgi::Output * out, std::string& choice) throw (xgi::exception::Exception);
  void createAll();

  void configureCustomRequest(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  void performAction();
  void readConfiguration(int& freq_tkmap, int& freq_sum);
  void setupQTests();

  SiPixelActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiPixelActionType flag) {theActionFlag = flag;}
   
private:
  SiPixelActionType theActionFlag;
  SiPixelActionExecutor* actionExecutor_;

  std::vector<std::string> tkMapOptions_;
  
protected:

};

#endif
