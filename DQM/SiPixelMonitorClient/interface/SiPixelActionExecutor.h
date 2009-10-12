#ifndef _SiPixelActionExecutor_h_
#define _SiPixelActionExecutor_h_

#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiPixelActionExecutor {

 public:

  SiPixelActionExecutor(            bool                           offlineXMLfile,
                                    bool                           Tier0Flag);
 ~SiPixelActionExecutor();

 void createSummary(    	    DQMStore    		 * bei);
 void bookOccupancyPlots(    	    DQMStore    		 * bei,
                                    bool                           hiRes,
									bool				isbarrel);
 void bookOccupancyPlots(    	    DQMStore    		 * bei,
                                    bool                           hiRes);
 void createOccupancy(    	    DQMStore    		 * bei);
 void setupQTests(      	    DQMStore    		 * bei);
 void checkQTestResults(	    DQMStore    		 * bei);
 void createTkMap(      	    DQMStore    		 * bei, 
                        	    std::string 	    	   mEName,
				    std::string 	    	   theTKType);
 bool readConfiguration(	    int 			 & tkmap_freq, 
                        	    int 			 & sum_barrel_freq, 
				    int 			 & sum_endcap_freq, 
				    int 			 & sum_grandbarrel_freq, 
				    int 			 & sum_grandendcap_freq,
				    int 			 & message_limit,
                                    int                          & source_type,
				    int                          & calib_type);
 bool readConfiguration(	    int 			 & tkmap_freq, 
                        	    int 			 & summary_freq);
 void readConfiguration(	    );
 void createLayout(     	    DQMStore    		 * bei);
 void fillLayout(       	    DQMStore    		 * bei);
 int getTkMapMENames(               std::vector<std::string>	 & names);
 void dumpModIds(                   DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 void dumpBarrelModIds(             DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 void dumpEndcapModIds(             DQMStore     		 * bei,
                                    edm::EventSetup const        & eSetup);
 


private:
  
  
  MonitorElement* getSummaryME(     DQMStore     		 * bei, 
                                    std::string 	     	   me_name);
  MonitorElement* getFEDSummaryME(  DQMStore     		 * bei, 
                                    std::string 	     	   me_name);
  void GetBladeSubdirs(DQMStore* bei, std::vector<std::string>& blade_subdirs); 
   void fillSummary(           	    DQMStore     		 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names,
				    bool isbarrel);
  void fillFEDErrorSummary(         DQMStore     		 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  void fillGrandBarrelSummaryHistos(DQMStore     		 * bei, 
			            std::vector<std::string> 	 & me_names);
  void fillGrandEndcapSummaryHistos(DQMStore     		 * bei, 
			            std::vector<std::string> 	 & me_names);
  void getGrandSummaryME(           DQMStore     		 * bei,
                                    int                      	   nbin, 
                                    std::string              	 & me_name, 
				    std::vector<MonitorElement*> & mes);
 
  void fillOccupancy(    	    DQMStore    		 * bei,
				    bool isbarrel);

  SiPixelConfigParser* configParser_;
  SiPixelConfigWriter* configWriter_;
  edm::ESHandle<SiPixelFedCablingMap> theCablingMap;
  
  std::vector<std::string> summaryMENames;
  std::vector<std::string> tkMapMENames;
  
  int message_limit_;
  int source_type_;
  int calib_type_;  
  int ndet_;
  bool offlineXMLfile_;
  bool Tier0Flag_;
  
  QTestHandle* qtHandler_;
  
  MonitorElement * OccupancyMap;
  MonitorElement * PixelOccupancyMap;
  
  TH2F * temp_H;
  TH2F * temp_1x2;
  TH2F * temp_1x5;
  TH2F * temp_2x3;
  TH2F * temp_2x4;
  TH2F * temp_2x5;
  
};
#endif
