#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiStripSummaryCreator;
class DaqMonitorBEInterface;
class MonitorUserInterface;

class SiStripActionExecutor {

 public:

  SiStripActionExecutor();
  virtual ~SiStripActionExecutor();

 void setupQTests(DaqMonitorBEInterface* bei);
 void createTkMap(DaqMonitorBEInterface* bei);
 bool readConfiguration(int& sum_freq);
 bool readConfiguration();
 void saveMEs(DaqMonitorBEInterface * bei, std::string fname);
 int getTkMapMENames(std::vector<std::string>& names);
 void createSummary(DaqMonitorBEInterface* bei);

 private:
 //  SiStripConfigParser* configParser_;
  std::vector<std::string> tkMapMENames;

  QTestHandle* qtHandler_;

  SiStripSummaryCreator* summaryCreator_;
};
#endif
