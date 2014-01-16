#ifndef _SiPixelWebInterface_h_
#define _SiPixelWebInterface_h_

#include "DQM/TrackerCommon/interface/WebInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"

class DQMStore;
class SiPixelActionExecutor;
class SiPixelInformationExtractor;
class SiPixelEDAClient;

class SiPixelWebInterface
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
			 CreatePlots              = 10,
			 ComputeGlobalQualityFlag = 11,
			 dumpModIds               = 12,
			 Occupancy    	          = 13};
			 
  SiPixelWebInterface(DQMStore* bei,
                      bool      offlineXMLfile,
		      bool      Tier0Flag);
  ~SiPixelWebInterface();

  void handleEDARequest(xgi::Input* in,
                        xgi::Output* out, 
			int niter,
                        bool isUpgrade);  
  
  SiPixelActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiPixelActionType flag) {theActionFlag = flag;}
  void performAction(bool isUpgrade);

  bool readConfiguration(int& tkmap_freq, 
                         int& summary_freq);
     
  bool createTkMap();
  void periodicTkMapUpdate( xgi::Output * out);
   
private:

  std::string get_from_multimap(std::multimap<std::string, std::string> &mymap, std::string key);

  SiPixelActionType theActionFlag;
  SiPixelActionExecutor* actionExecutor_;
  SiPixelInformationExtractor* infoExtractor_;

  void returnReplyXml(xgi::Output * out, 
                      const std::string& name, 
		      const std::string& comment);

  std::vector<std::string> tkMapOptions_;
  bool tkMapCreated;
  std::multimap<std::string, std::string> requestMap_;
  DQMStore* bei_;
  bool offlineXMLfile_;
  bool Tier0Flag_;

protected:

};

#endif
