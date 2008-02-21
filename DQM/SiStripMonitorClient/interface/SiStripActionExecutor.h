#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiStripSummaryCreator;
class DaqMonitorBEInterface;
class MonitorUserInterface;
class SiStripTrackerMapCreator;

class SiStripActionExecutor {

 public:

  SiStripActionExecutor();
  virtual ~SiStripActionExecutor();


 bool readConfiguration();
 bool readConfiguration(int& sum_freq);
 bool readTkMapConfiguration();

 void saveMEs(DaqMonitorBEInterface * bei, std::string fname);
 void createSummary(DaqMonitorBEInterface* bei);
 void createTkMap(const edm::ParameterSet & tkmapPset, 
		  const edm::ESHandle<SiStripFedCabling>& fedcabling, DaqMonitorBEInterface* bei);
 private:

  std::vector<std::string> tkMapMENames;

  SiStripSummaryCreator* summaryCreator_;
  SiStripTrackerMapCreator* tkMapCreator_;
};
#endif
