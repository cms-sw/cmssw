#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiStripActionExecutor {

 public:
  typedef std::map<int,std::vector <std::pair <int,float> > > DetMapType;

  SiStripActionExecutor();
 ~SiStripActionExecutor();

 void createSummary(MonitorUserInterface* mui);

 void setupQTests(MonitorUserInterface * mui);
 void checkQTestResults(MonitorUserInterface * mui);
 void createCollation(MonitorUserInterface * mui);
 void createTkMap(MonitorUserInterface* mui);
 bool readConfiguration(int& tkmap_freq, int& sum_freq);
 void readConfiguration();
 void createLayout(MonitorUserInterface * mui);
 void fillLayout(MonitorUserInterface * mui);
 void saveMEs(MonitorUserInterface * mui, std::string fname);


 private:
  MonitorElement* getSummaryME(MonitorUserInterface* mui, std::string me_name);
  void getValuesForTkMap(MonitorUserInterface* mui,
     std::vector<std::string> me_names, SiStripActionExecutor::DetMapType& values);
     void fillSummary(MonitorUserInterface* mui, std::string dir_name,
     std::vector<std::string>& me_names);
  void drawMEs(int idet, std::vector<MonitorElement*>& mon_elements, 
                    std::vector<std::pair <int, float> > & values);
  void fillGrandSummaryHistos(MonitorUserInterface* mui, 
			      std::vector<std::string>& me_names);
  void getGrandSummaryME(MonitorUserInterface* mui,int nbin, 
      std::string& me_name, std::vector<MonitorElement*> & mes);

  SiStripConfigParser* configParser_;
  SiStripConfigWriter* configWriter_;
};
#endif
