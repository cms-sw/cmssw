#ifndef _SiPixelWebInterface_h_
#define _SiPixelWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/Core/interface/MonitorElement.h"
class SiPixelActionExecutor;
class SiPixelInformationExtractor;
class SiPixelEDAClient;

class SiPixelWebInterface : public WebInterface
{

public:

  enum SiPixelActionType{NoAction     	          =  0,
			 Summary    	          =  1,
                         setupQTest  	          =  2,
			 QTestResult 	          =  3,
			 CreateTkMap 	          =  4,
                         PlotSingleModuleHistos   =  5,
                         PlotSingleHistogram      =  6,
			 PlotTkMapHistogram       =  7,
			 periodicTrackerMapUpdate =  8,
			 PlotHistogramFromPath    =  9,
			 ComputeGlobalQualityFlag = 10,
			 dumpModIds               = 11};

  SiPixelWebInterface(std::string theContextURL, 
                      std::string theApplicationURL, 
		      DQMOldReceiver ** _mui_p);
  ~SiPixelWebInterface();

  void handleEDARequest(xgi::Input* in,
                        xgi::Output* out, 
			int niter);  
  
  void createAll();

  void performAction();
  void readConfiguration(int& freq_tkmap, 
                         int& freq_barrel_sum, 
			 int& freq_endcap_sum, 
			 int& freq_grandbarrel_sum, 
			 int& freq_grandendcap_sum, 
			 int& message_limit,
			 int& source_type);
  void readConfiguration(int& tkmap_freq, 
                         int& summary_freq);
  
  void setupQTests();

  SiPixelActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiPixelActionType flag) {theActionFlag = flag;}
   
  bool createTkMap();
  void periodicTkMapUpdate( xgi::Output * out);
   
  float qflag_;
  float returnQFlag() {return qflag_;}
  
private:

  SiPixelActionType theActionFlag;
  SiPixelActionExecutor* actionExecutor_;
  SiPixelInformationExtractor* infoExtractor_;

  void returnReplyXml(xgi::Output * out, 
                      const std::string& name, 
		      const std::string& comment);

  std::vector<std::string> tkMapOptions_;
  bool tkMapCreated;
  std::multimap<std::string, std::string> requestMap_;
  xgi::Output * theOut ;
  std::string fileName_;  
  
  int allMods_;
  int errorMods_;
  
protected:

};

#endif
