#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiStripSummaryCreator;

class SiStripActionExecutor {

 public:

  SiStripActionExecutor();
  virtual ~SiStripActionExecutor();

 void setupQTests(MonitorUserInterface * mui);
 void createCollation(MonitorUserInterface * mui);
 void createTkMap(MonitorUserInterface* mui);
 bool readConfiguration(int& sum_freq);
 void readConfiguration();
 void saveMEs(MonitorUserInterface * mui, std::string fname);
 bool getCollationFlag(){return collationDone;}
 int getTkMapMENames(std::vector<std::string>& names);
 void createSummary(MonitorUserInterface* mui);

 private:
 //  SiStripConfigParser* configParser_;
  std::vector<std::string> tkMapMENames;
  bool collationDone;

  QTestHandle* qtHandler_;

  SiStripSummaryCreator* summaryCreator_;
};
#endif
