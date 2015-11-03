#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <TTree.h>

class SiStripSummaryCreator;
class DQMStore;
class MonitorUserInterface;
class SiStripTrackerMapCreator;
class SiStripQualityChecker;
class SiStripFedCabling;
class SiStripDetCabling;
class SiStripConfigWriter;
class SiStripDetInfoFileReader;

class SiStripActionExecutor {

 public:

  SiStripActionExecutor(edm::ParameterSet const& ps);
  virtual ~SiStripActionExecutor();


 bool readConfiguration();
 // bool readTkMapConfiguration();
 bool readTkMapConfiguration(const edm::EventSetup& eSetup);

 void saveMEs(DQMStore * dqm_store, std::string fname);
 void createSummary(DQMStore* dqm_store);
 void createSummaryOffline(DQMStore* dqm_store);
 void createTkMap(const edm::ParameterSet & tkmapPset, 
                  DQMStore* dqm_store, std::string& map_type, const edm::EventSetup& eSetup);
 void createOfflineTkMap(const edm::ParameterSet & tkmapPset,
			 DQMStore* dqm_store, std::string& map_type, const edm::EventSetup& eSetup);
 void createTkInfoFile(std::vector<std::string> tkhmap_names, TTree* tkinfo_tree, DQMStore* dqm_store);

 void createStatus(DQMStore* dqm_store);
 void fillDummyStatus();
 void fillStatus(DQMStore* dqm_store, const edm::ESHandle<SiStripDetCabling>& fedcabling, const edm::EventSetup& eSetup);
 void fillStatusAtLumi(DQMStore* dqm_store);

 void createDummyShiftReport();
 void createShiftReport(DQMStore * dqm_store);
 void printReportSummary(MonitorElement* me, std::ostringstream& str_val, std::string name);
 void printShiftHistoParameters(DQMStore * dqm_store,
             std::map<std::string, std::vector<std::string> >&layout_map,std::ostringstream& str_val);
 void printFaultyModuleList(DQMStore * dqm_store, std::ostringstream& str_val);
 void createFaultyModuleMEs(DQMStore *dqm_store);

 private:

  std::vector<std::string> tkMapMENames;

  SiStripSummaryCreator* summaryCreator_;
  SiStripTrackerMapCreator* tkMapCreator_;
  SiStripQualityChecker*   qualityChecker_;

  SiStripConfigWriter* configWriter_;

  SiStripDetInfoFileReader* detInfoFileReader_;

  edm::ParameterSet pSet_;

};
#endif
