#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

class SiStripSummaryCreator;
class DQMStore;
class MonitorUserInterface;
class SiStripTrackerMapCreator;
class SiStripFedCabling;
class SiStripDetCabling;
class SiStripConfigWriter;

class SiStripActionExecutor {

 public:

  SiStripActionExecutor();
  virtual ~SiStripActionExecutor();


 bool readConfiguration();
 bool readConfiguration(int& sum_freq);
 bool readTkMapConfiguration();

 void saveMEs(DQMStore * dqm_store, std::string fname);
 void createSummary(DQMStore* dqm_store);
 void createSummaryOffline(DQMStore* dqm_store);
 void createTkMap(const edm::ParameterSet & tkmapPset, 
		  const edm::ESHandle<SiStripFedCabling>& fedcabling, DQMStore* dqm_store);

 void bookGlobalStatus(DQMStore* dqm_store);     
 void fillGlobalStatusFromModule(DQMStore* dqm_store);
 void fillGlobalStatusFromLayer(DQMStore* dqm_store);
 
 void resetGlobalStatus();
 void fillDummyGlobalStatus();
 void createDummyShiftReport();
 void createShiftReport(DQMStore * dqm_store);
 void printReportSummary(MonitorElement* me, std::ostringstream& str_val, std::string name);
 void printShiftHistoParameters(DQMStore * dqm_store,
             std::map<std::string, std::vector<std::string> >&layout_map,std::ostringstream& str_val);

 private:

 void fillSubDetStatusFromModule(DQMStore* dqm_store, std::string& dname, int& tot_me_subdet,
		       int& error_me_subdet, unsigned int xbin);
 void fillSubDetStatusFromLayer(DQMStore* dqm_store, std::string& dname, int& tot_me_subdet,
		       int& error_me_subdet, unsigned int xbin);
  void fillClusterReport(DQMStore* dqm_store, std::string& dname, int xbin);
 bool goToDir(DQMStore * dqm_store, std::string name);


  std::vector<std::string> tkMapMENames;

  SiStripSummaryCreator* summaryCreator_;
  SiStripTrackerMapCreator* tkMapCreator_;

  MonitorElement * SummaryReport;
  MonitorElement * SummaryReportMap;
  MonitorElement * SummaryTIB;
  MonitorElement * SummaryTOB;
  MonitorElement * SummaryTIDF;
  MonitorElement * SummaryTIDB;
  MonitorElement * SummaryTECF;
  MonitorElement * SummaryTECB;

  MonitorElement * OnTrackClusterReport;

  bool bookedGlobalStatus_;
  SiStripConfigWriter* configWriter_;
};
#endif
