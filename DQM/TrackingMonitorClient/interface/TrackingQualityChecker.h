#ifndef _TrackingQualityChecker_h_
#define _TrackingQualityChecker_h_

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

class TrackingDetCabling;

class TrackingQualityChecker {
public:
  typedef dqm::harvesting::DQMStore DQMStore;
  typedef dqm::harvesting::MonitorElement MonitorElement;

  TrackingQualityChecker(edm::ParameterSet const& ps);
  virtual ~TrackingQualityChecker();

  void bookGlobalStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void bookLSStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void resetGlobalStatus();
  void resetLSStatus();
  void fillDummyGlobalStatus();
  void fillDummyLSStatus();
  void fillGlobalStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void fillLSStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);

private:
  struct TrackingMEs {
    MonitorElement* TrackingFlag;
    std::string HistoDir;
    std::string HistoName;
  };

  struct TrackingLSMEs {
    MonitorElement* TrackingFlag;
    std::string HistoLSDir;
    std::string HistoLSName;
    float HistoLSLowerCut;
    float HistoLSUpperCut;
  };

  void fillTrackingStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void fillTrackingStatusAtLumi(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);

  void fillStatusHistogram(MonitorElement*, int xbin, int ybin, float val);

  std::map<std::string, TrackingMEs> TrackingMEsMap;
  std::map<std::string, TrackingLSMEs> TrackingLSMEsMap;

  MonitorElement* TrackGlobalSummaryReportMap;
  MonitorElement* TrackGlobalSummaryReportGlobal;

  MonitorElement* TrackLSSummaryReportGlobal;

  edm::ParameterSet pSet_;
  bool verbose_;

  bool bookedTrackingGlobalStatus_;
  bool bookedTrackingLSStatus_;

  std::string TopFolderName_;
};
#endif
