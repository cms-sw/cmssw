#ifndef _SiPixelActionExecutor_h_
#define _SiPixelActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiPixelActionExecutor {

 public:
  typedef std::map<int,std::vector <std::pair <int,float> > > DetMapType;

  SiPixelActionExecutor();
 ~SiPixelActionExecutor();

 void createSummary(MonitorUserInterface* mui);

 void setupQTests(MonitorUserInterface * mui);
 void checkQTestResults(MonitorUserInterface * mui);
 void createCollation(MonitorUserInterface * mui);
// void createTkMap(MonitorUserInterface* mui);
 bool readConfiguration(int& tkmap_freq, int& sum_barrel_freq, int& sum_endcap_freq);
 void readConfiguration();
 void createLayout(MonitorUserInterface * mui);
 void fillLayout(MonitorUserInterface * mui);
 void saveMEs(MonitorUserInterface * mui, std::string fname);
 bool getCollationFlag(){return collationDone;}
// int getTkMapMENames(std::vector<std::string>& names);

 private:
  MonitorElement* getSummaryME(MonitorUserInterface* mui, std::string me_name);
  void fillBarrelSummary(MonitorUserInterface* mui, std::string dir_name,
     std::vector<std::string>& me_names);
  void fillEndcapSummary(MonitorUserInterface* mui, std::string dir_name,
     std::vector<std::string>& me_names);
  void drawMEs(int idet, std::vector<MonitorElement*>& mon_elements, 
                    std::vector<std::pair <int, float> > & values);
  void fillGrandBarrelSummaryHistos(MonitorUserInterface* mui, 
			      std::vector<std::string>& me_names);
  void fillGrandEndcapSummaryHistos(MonitorUserInterface* mui, 
			      std::vector<std::string>& me_names);
  void getGrandSummaryME(MonitorUserInterface* mui,int nbin, 
      std::string& me_name, std::vector<MonitorElement*> & mes);


  SiPixelConfigParser* configParser_;
  SiPixelConfigWriter* configWriter_;
  
  std::vector<std::string> summaryMENames;
//  std::vector<std::string> tkMapMENames;
  bool collationDone;
};
#endif
