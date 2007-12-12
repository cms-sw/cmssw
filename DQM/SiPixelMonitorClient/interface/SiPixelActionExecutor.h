#ifndef _SiPixelActionExecutor_h_
#define _SiPixelActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

using namespace std ; 

class SiPixelActionExecutor {

 public:

  SiPixelActionExecutor();
 ~SiPixelActionExecutor();

 void createSummary(    	    DaqMonitorBEInterface    	 * bei);

 void setupQTests(      	    DaqMonitorBEInterface    	 * bei);
 void checkQTestResults(	    DaqMonitorBEInterface    	 * bei);
 void createTkMap(      	    DaqMonitorBEInterface    	 * bei, 
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
 void createLayout(     	    DaqMonitorBEInterface    	 * bei);
 void fillLayout(       	    DaqMonitorBEInterface    	 * bei);
 int getTkMapMENames(               std::vector<std::string>	 & names);

 private:
  MonitorElement* getSummaryME(     DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   me_name);
  MonitorElement* getFEDSummaryME(  DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   me_name);
  void fillBarrelSummary(           DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  void fillEndcapSummary(           DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  void fillFEDErrorSummary(         DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  void fillGrandBarrelSummaryHistos(DaqMonitorBEInterface     	 * bei, 
			            std::vector<std::string> 	 & me_names);
  void fillGrandEndcapSummaryHistos(DaqMonitorBEInterface     	 * bei, 
			            std::vector<std::string> 	 & me_names);
  void getGrandSummaryME(           DaqMonitorBEInterface     	 * bei,
                                    int                      	   nbin, 
                                    std::string              	 & me_name, 
				    std::vector<MonitorElement*> & mes);


  SiPixelConfigParser* configParser_;
  SiPixelConfigWriter* configWriter_;
  
  std::vector<std::string> summaryMENames;
  std::vector<std::string> tkMapMENames;
  
  int message_limit_;
  int source_type_;
  
  QTestHandle* qtHandler_;
  
};
#endif
