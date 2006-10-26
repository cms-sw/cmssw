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
 MonitorElement* getSummaryME(MonitorUserInterface* mui, std::string& name, int nval);
  void getValuesForTkMap(MonitorUserInterface* mui,
     std::vector<std::string> me_names, SiStripActionExecutor::DetMapType& values);
  void fillSummary(MonitorUserInterface* mui);
  void drawMEs(int idet, std::vector<MonitorElement*>& mon_elements, 
                    std::vector<std::pair <int, float> > & values);
  void fillGrandSummaryHistos(MonitorUserInterface* mui);
  void fillSummaryHistos(MonitorUserInterface* mui);

  SiStripConfigParser* configParser_;
  SiStripConfigWriter* configWriter_;
  std::vector<std::string> summaryMENames;
};
#endif
