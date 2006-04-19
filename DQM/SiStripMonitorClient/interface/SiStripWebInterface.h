#ifndef _SiStripWebInterface_h_
#define _SiStripWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class SiStripActionExecutor;
class SiStripQualityTester;

class SiStripWebInterface : public WebInterface
{
 public:
 
  SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);
 ~SiStripWebInterface();

  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);

  void subscribeAll(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  void setupQTest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  void createTkMap(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  void createSummary(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  void saveToFile(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);

  int getUpdates();

 private:

  // the SiStripQualityTester
  SiStripQualityTester* theQualityTester;
  // the SiStripActionActionExecutor
  SiStripActionExecutor* theActionExecutor;

 protected:


};

#endif
