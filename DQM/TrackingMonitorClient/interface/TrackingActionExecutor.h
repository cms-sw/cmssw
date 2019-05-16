#ifndef _TrackingActionExecutor_h_
#define _TrackingActionExecutor_h_

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

class SiStripSummaryCreator;
class MonitorUserInterface;
class SiStripTrackerMapCreator;
class TrackingQualityChecker;
class SiStripFedCabling;
class SiStripDetCabling;
class SiStripConfigWriter;

class TrackingActionExecutor {
public:
  TrackingActionExecutor(edm::ParameterSet const& ps);
  virtual ~TrackingActionExecutor();

  void createGlobalStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void createLSStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void fillDummyGlobalStatus();
  void fillDummyLSStatus();
  void fillGlobalStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void fillStatusAtLumi(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);

  void createDummyShiftReport();
  void createShiftReport(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void printReportSummary(MonitorElement* me, std::ostringstream& str_val, std::string name);
  void printShiftHistoParameters(DQMStore::IBooker& ibooker,
                                 DQMStore::IGetter& igetter,
                                 std::map<std::string, std::vector<std::string> >& layout_map,
                                 std::ostringstream& str_val);

private:
  std::vector<std::string> tkMapMENames;

  TrackingQualityChecker* qualityChecker_;

  SiStripConfigWriter* configWriter_;

  edm::ParameterSet pSet_;
};
#endif
