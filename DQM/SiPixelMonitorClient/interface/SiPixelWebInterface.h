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
                         setupQTest  	          =  3,
			 QTestResult 	          =  4,
			 CreateTkMap 	          =  5,
                         PlotSingleModuleHistos   =  6,
                         PlotSingleHistogram      =  7,
			 PlotTkMapHistogram       =  8,
			 periodicTrackerMapUpdate =  9,
			 PlotHistogramFromPath    = 10,
			 PlotHistogramFromLayout     = 11,
			 PlotErrorOverviewHistogram  = 12};

  SiPixelWebInterface(std::string theContextURL, 
                      std::string theApplicationURL, 
		      MonitorUserInterface ** _mui_p);
  ~SiPixelWebInterface();

  void handleCustomRequest(xgi::Input * in, 
                           xgi::Output * out ) 
			   throw (xgi::exception::Exception);
  void handleEDARequest(xgi::Input* in,
                        xgi::Output* out, 
			int niter);  
  
  void createAll();

  void configureCustomRequest(xgi::Input * in, 
                              xgi::Output * out) 
			      throw (xgi::exception::Exception);
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

  void setOutputFileName(std::string fname){fileName_ = fname;};

  SiPixelActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiPixelActionType flag) {theActionFlag = flag;}
   
  bool createTkMap();
  void periodicTkMapUpdate( xgi::Output * out);
   
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
  
  
protected:

};

#endif
