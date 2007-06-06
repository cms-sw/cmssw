#ifndef _SiPixelWebInterface_h_
#define _SiPixelWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/Core/interface/MonitorElement.h"
class SiPixelActionExecutor;
class SiPixelInformationExtractor;

class SiPixelWebInterface : public WebInterface
{

public:

  enum SiPixelActionType{NoAction     	          =  0,
                         SubscribeAll 	          =  1,
			 Summary    	          =  2,
			 Collate    	          =  3,
                         setupQTest  	          =  4,
			 QTestResult 	          =  5,
			 CreateTkMap 	          =  6,
                         SaveData     	          =  7,
                         PlotSingleModuleHistos   =  8,
                         PlotSingleHistogram      =  9,
			 PlotTkMapHistogram       = 10,
			 periodicTrackerMapUpdate = 11};

  SiPixelWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p);
  ~SiPixelWebInterface();

  void handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  void createAll();

  void configureCustomRequest(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  void performAction();
//  void readConfiguration(int& freq_tkmap, int& freq_sum);
  void readConfiguration(int& freq_tkmap, int& freq_barrel_sum, int& freq_endcap_sum);
  void setupQTests();

  SiPixelActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiPixelActionType flag) {theActionFlag = flag;}
   
  bool createTkMap();
  void periodicTkMapUpdate( xgi::Output * out);
   
private:

  SiPixelActionType theActionFlag;
  SiPixelActionExecutor* actionExecutor_;
  SiPixelInformationExtractor* infoExtractor_;

  void returnReplyXml(xgi::Output * out, const std::string& name, const std::string& comment);

  std::vector<std::string> tkMapOptions_;
  bool tkMapCreated;
  std::multimap<std::string, std::string> requestMap_;
  xgi::Output * theOut ;
  
  
protected:

};

#endif
