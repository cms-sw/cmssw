#ifndef _SiPixelActionExecutor_h_
#define _SiPixelActionExecutor_h_

#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiPixelActionExecutor {

 public:

  SiPixelActionExecutor();
 ~SiPixelActionExecutor();

 void createSummary(    	    DQMStore    		 * bei);

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
                                    int                          & source_type);
 bool readConfiguration(	    int 			 & tkmap_freq, 
                        	    int 			 & summary_freq);
 void readConfiguration(	    );
 void createLayout(     	    DQMStore    		 * bei);
 void fillLayout(       	    DQMStore    		 * bei);
 int getTkMapMENames(               std::vector<std::string>	 & names);
 void dumpModIds(                   DQMStore     		 * bei);
 void dumpBarrelModIds(             DQMStore     		 * bei);
 void dumpEndcapModIds(             DQMStore     		 * bei);
 


private:
  
  
  MonitorElement* getSummaryME(     DQMStore     		 * bei, 
                                    std::string 	     	   me_name);
  MonitorElement* getFEDSummaryME(  DQMStore     		 * bei, 
                                    std::string 	     	   me_name);
  void fillBarrelSummary(           DQMStore     		 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  void fillEndcapSummary(           DQMStore     		 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
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

  SiPixelConfigParser* configParser_;
  SiPixelConfigWriter* configWriter_;
  
  std::vector<std::string> summaryMENames;
  std::vector<std::string> tkMapMENames;
  
  int message_limit_;
  int source_type_;
  
  int ndet_;
  
  QTestHandle* qtHandler_;
  
};
#endif
