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
                         SaveData=4, 
                         PlotSingleModuleHistos=5, PlotGlobalHistos=6,
                         PlotHistogramFromPath=7, PlotTkMapHistogram=8,
                         PlotHistogramFromLayout=9};

  SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);
 ~SiStripWebInterface();

  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  
  void createAll();
 
  void configureCustomRequest(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
    void handleAnalyserRequest(xgi::Input* in,xgi::Output* out, int niter);

  void performAction();
  void readConfiguration(int& freq_sum);
  void setupQTests();

  SiStripActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiStripActionType flag) {theActionFlag = flag;}

  void setOutputFileName(std::string name) {fileName_ = name;}
  void setTkMapFlag(bool flg) {tkMapCreated = flg;} 
      
  private:

  SiStripActionType theActionFlag;
  SiStripActionExecutorQTest* actionExecutor_;
  SiStripInformationExtractor* infoExtractor_;

  void returnReplyXml(xgi::Output * out, const std::string& name, const std::string& comment);

  bool tkMapCreated;
  std::multimap<std::string, std::string> requestMap_;

  std::string fileName_;
 protected:


};

#endif
