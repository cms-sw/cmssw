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

 //void createSummary(    	    MonitorUserInterface    	 * mui);
 void createSummary(    	    DaqMonitorBEInterface    	 * bei);

 //void setupQTests(      	    MonitorUserInterface    	 * mui);
 void setupQTests(      	    DaqMonitorBEInterface    	 * bei);
 //void checkQTestResults(	    MonitorUserInterface    	 * mui);
 void checkQTestResults(	    DaqMonitorBEInterface    	 * bei);
 void createCollation(  	    MonitorUserInterface    	 * mui);
 //void createTkMap(      	    MonitorUserInterface    	 * mui, 
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
 void readConfiguration(	    );
 //void createLayout(     	    MonitorUserInterface    	 * mui);
 void createLayout(     	    DaqMonitorBEInterface    	 * bei);
 //void fillLayout(       	    MonitorUserInterface    	 * mui);
 void fillLayout(       	    DaqMonitorBEInterface    	 * bei);
 //void saveMEs(          	    MonitorUserInterface    	 * mui, 
 void saveMEs(          	    DaqMonitorBEInterface    	 * bei, 
                        	    std::string 	    	   fname);
 int getTkMapMENames(               std::vector<std::string>	 & names);
 bool getCollationFlag(){return collationDone;}

 private:
  //MonitorElement* getSummaryME(     MonitorUserInterface     	 * mui, 
  MonitorElement* getSummaryME(     DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   me_name);
  //MonitorElement* getFEDSummaryME(  MonitorUserInterface     	 * mui, 
  MonitorElement* getFEDSummaryME(  DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   me_name);
  //void fillBarrelSummary(           MonitorUserInterface     	 * mui, 
  void fillBarrelSummary(           DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  //void fillEndcapSummary(           MonitorUserInterface     	 * mui, 
  void fillEndcapSummary(           DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  //void fillFEDErrorSummary(         MonitorUserInterface     	 * mui, 
  void fillFEDErrorSummary(         DaqMonitorBEInterface     	 * bei, 
                                    std::string 	     	   dir_name,
                                    std::vector<std::string> 	 & me_names);
  //void fillGrandBarrelSummaryHistos(MonitorUserInterface     	 * mui, 
  void fillGrandBarrelSummaryHistos(DaqMonitorBEInterface     	 * bei, 
			            std::vector<std::string> 	 & me_names);
  //void fillGrandEndcapSummaryHistos(MonitorUserInterface     	 * mui, 
  void fillGrandEndcapSummaryHistos(DaqMonitorBEInterface     	 * bei, 
			            std::vector<std::string> 	 & me_names);
  //void getGrandSummaryME(           MonitorUserInterface     	 * mui,
  void getGrandSummaryME(           DaqMonitorBEInterface     	 * bei,
                                    int                      	   nbin, 
                                    std::string              	 & me_name, 
				    std::vector<MonitorElement*> & mes);


  SiPixelConfigParser* configParser_;
  SiPixelConfigWriter* configWriter_;
  
  std::vector<std::string> summaryMENames;
  std::vector<std::string> tkMapMENames;
  
  bool collationDone;
  
  int message_limit_;
  int source_type_;
  
  QTestHandle* qtHandler_;
  
};
#endif
