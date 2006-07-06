#ifndef _SiStripWebInterface_h_
#define _SiStripWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/Core/interface/MonitorElement.h"
class SiStripActionExecutor;

class SiStripWebInterface : public WebInterface
{
 public:
 
  enum SiStripActionType{NoAction=0, SubscribeAll=1, Summary=2, Collate=3,
                         QTestResult=4, PersistantTkMap=5, 
                         TemporaryTkMap=6, SaveData=7};

  SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);
 ~SiStripWebInterface();

  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
 void readSelectedRequest(xgi::Input * in, xgi::Output * out, std::string& choice) throw (xgi::exception::Exception);
  void createAll();
 
  void configureCustomRequest(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  void performAction();
  void readConfiguration(int& freq_tkmap, int& freq_sum);
  void setupQTests();

  SiStripActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiStripActionType flag) {theActionFlag = flag;}
   
  private:

  SiStripActionType theActionFlag;
  SiStripActionExecutor* actionExecutor_;

  std::vector<std::string> tkMapOptions_;
 protected:


};

#endif
