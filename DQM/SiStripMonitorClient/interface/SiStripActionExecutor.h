#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
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
class SiStripFedCabling;
class SiStripDetCabling;

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

 void bookGlobalStatus(DQMStore* dqm_store);     
 void fillGlobalStatus(const edm::ESHandle<SiStripDetCabling>& detcabling, DQMStore* dqm_store);
 void resetGlobalStatus();

 private:

 void fillSubDetStatus(DQMStore* dqm_store, std::string& dname, int& tot_me_subdet,
		       int& error_me_subdet,int xbin);


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

  bool bookedGlobalStatus_;
};
#endif
