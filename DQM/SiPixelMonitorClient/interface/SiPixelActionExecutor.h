#ifndef _SiPixelActionExecutor_h_
#define _SiPixelActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
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

 void createSummary(MonitorUserInterface* mui);

 void setupQTests(MonitorUserInterface * mui);
 void checkQTestResults(MonitorUserInterface * mui);
 void createCollation(MonitorUserInterface * mui);
 void createTkMap(MonitorUserInterface* mui, std::string mEName);
 bool readConfiguration(int& tkmap_freq, int& sum_barrel_freq, int& sum_endcap_freq, int& sum_grandbarrel_freq, int& sum_grandendcap_freq);
 void readConfiguration();
 void createLayout(MonitorUserInterface * mui);
 void fillLayout(MonitorUserInterface * mui);
 void saveMEs(MonitorUserInterface * mui, std::string fname);
 bool getCollationFlag(){return collationDone;}
 int getTkMapMENames(std::vector<std::string>& names);

 private:
  MonitorElement* getSummaryME(MonitorUserInterface* mui, std::string me_name);
  void fillBarrelSummary(MonitorUserInterface* mui, std::string dir_name,
     std::vector<std::string>& me_names);
  void fillEndcapSummary(MonitorUserInterface* mui, std::string dir_name,
     std::vector<std::string>& me_names);
  void fillGrandBarrelSummaryHistos(MonitorUserInterface* mui, 
			      std::vector<std::string>& me_names);
  void fillGrandEndcapSummaryHistos(MonitorUserInterface* mui, 
			      std::vector<std::string>& me_names);
  void getGrandSummaryME(MonitorUserInterface* mui,int nbin, 
      std::string& me_name, std::vector<MonitorElement*> & mes);


  SiPixelConfigParser* configParser_;
  SiPixelConfigWriter* configWriter_;
  
  std::vector<std::string> summaryMENames;
  std::vector<std::string> tkMapMENames;
  
  bool collationDone;
  
  QTestHandle* qtHandler_;
  
};
#endif
