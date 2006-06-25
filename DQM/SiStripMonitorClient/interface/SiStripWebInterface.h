#ifndef _SiStripWebInterface_h_
#define _SiStripWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class SiStripWebInterface : public WebInterface
{
 public:
 
  enum SiStripActionType{NoAction=0, Summary=1, Collate=2, QTestResult=3, PersistantTkMap=4, 
                         TemporaryTkMap=5, SaveData=6};

  SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);
 ~SiStripWebInterface();

  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  void subscribeAll(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);

  SiStripActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiStripActionType flag) {theActionFlag = flag;}
  
   
  private:

  SiStripActionType theActionFlag;

 protected:


};

#endif
