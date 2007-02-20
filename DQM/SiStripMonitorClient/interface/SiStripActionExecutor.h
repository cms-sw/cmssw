#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiStripActionExecutor {

 public:

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
 bool getCollationFlag(){return collationDone;}
 int getTkMapMENames(std::vector<std::string>& names);

 private:
 MonitorElement* getSummaryME(MonitorUserInterface* mui,
                              std::string& name, bool ifl);
  void fillSummary(MonitorUserInterface* mui);

  void fillGrandSummaryHistos(MonitorUserInterface* mui);
  void fillSummaryHistos(MonitorUserInterface* mui);
  void fillHistos(int ival, int istep, MonitorElement* me_src, 
	                                 	  MonitorElement* me);
  SiStripConfigParser* configParser_;
  SiStripConfigWriter* configWriter_;
  std::vector<std::string> summaryMENames;
  std::vector<std::string> tkMapMENames;
  bool collationDone;

  QTestHandle* qtHandler_;
};
#endif
