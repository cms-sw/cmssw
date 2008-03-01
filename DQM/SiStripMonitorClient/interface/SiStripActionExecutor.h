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
class DQMStore;
class MonitorUserInterface;
class SiStripTrackerMapCreator;

class SiStripActionExecutor {

 public:

  SiStripActionExecutor();
  virtual ~SiStripActionExecutor();


 bool readConfiguration();
 bool readConfiguration(int& sum_freq);
 bool readTkMapConfiguration();

 void saveMEs(DQMStore * dqm_store, std::string fname);
 void createSummary(DQMStore* dqm_store);
 void createTkMap(const edm::ParameterSet & tkmapPset, 
		  const edm::ESHandle<SiStripFedCabling>& fedcabling, DQMStore* dqm_store);
 private:

  std::vector<std::string> tkMapMENames;

  SiStripSummaryCreator* summaryCreator_;
  SiStripTrackerMapCreator* tkMapCreator_;
};
#endif
