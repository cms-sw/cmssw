#ifndef _SiStripWebInterface_h_
#define _SiStripWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/Core/interface/MonitorElement.h"
class SiStripActionExecutorQTest;
class SiStripInformationExtractor;

class SiStripWebInterface : public WebInterface
{
 public:
 
  enum SiStripActionType{NoAction=0, SubscribeAll=1, Summary=2, Collate=3,
                         CreateTkMap=4, SaveData=5, 
                         PlotSingleModuleHistos=6, PlotGlobalHistos=7,
                         PlotHistogramFromPath=8, PlotTkMapHistogram=9,
                         PlotHistogramFromLayout=10};

  SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);
 ~SiStripWebInterface();

  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  void createAll();
 
  void configureCustomRequest(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  void performAction();
  void readConfiguration(int& freq_tkmap, int& freq_sum);
  void setupQTests();

  SiStripActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiStripActionType flag) {theActionFlag = flag;}

  bool createTkMap();
  void setOutputFileName(std::string name) {fileName_ = name;}
    
  private:

  SiStripActionType theActionFlag;
  SiStripActionExecutorQTest* actionExecutor_;
  SiStripInformationExtractor* infoExtractor_;

  void returnReplyXml(xgi::Output * out, const std::string& name, const std::string& comment);

  bool tkMapCreated;
  std::multimap<std::string, std::string> requestMap_;

  string fileName_;
 protected:


};

#endif
