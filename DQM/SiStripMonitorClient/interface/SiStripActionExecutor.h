#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

class SiStripSummaryCreator;
class MonitorUserInterface;
class SiStripTrackerMapCreator;
class SiStripQualityChecker;
class SiStripFedCabling;
class SiStripDetCabling;
class SiStripConfigWriter;

class SiStripActionExecutor {

 public:

  SiStripActionExecutor(edm::ParameterSet const& ps);
  virtual ~SiStripActionExecutor();


 bool readConfiguration();
 bool readTkMapConfiguration(const edm::EventSetup& eSetup);

 void saveMEs(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::string fname);
 void createSummary(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
 void createSummaryOffline(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
 void createTkMap(const edm::ParameterSet & tkmapPset, 
		  DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::string& map_type, edm::ESHandle<SiStripQuality> & ssq);
 void createOfflineTkMap(const edm::ParameterSet & tkmapPset,
			 DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::string& map_type, edm::ESHandle<SiStripQuality> & ssq);

 void createStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
 void fillDummyStatus();
 void fillStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const edm::ESHandle<SiStripDetCabling>& fedcabling, const TrackerTopology *tTopo);
 void fillStatusAtLumi(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);

 void createDummyShiftReport();
 void createShiftReport(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
 void printReportSummary(MonitorElement* me, std::ostringstream& str_val, std::string name);
 void printShiftHistoParameters(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
             std::map<std::string, std::vector<std::string> >&layout_map,std::ostringstream& str_val);
 void printFaultyModuleList(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::ostringstream& str_val);
 void createFaultyModuleMEs(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);

 private:

  std::vector<std::string> tkMapMENames;

  SiStripSummaryCreator* summaryCreator_;
  SiStripTrackerMapCreator* tkMapCreator_;
  SiStripQualityChecker*   qualityChecker_;

  SiStripConfigWriter* configWriter_;

  edm::ParameterSet pSet_;
};
#endif
