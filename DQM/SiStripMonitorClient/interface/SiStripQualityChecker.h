#ifndef _SiStripQualityChecker_h_
#define _SiStripQualityChecker_h_

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

class TkDetMap;
class TrackerTopology;
class SiStripDetCabling;

class SiStripQualityChecker {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripQualityChecker(edm::ParameterSet const& ps);
  ~SiStripQualityChecker();

  void bookStatus(DQMStore& dqm_store);
  void resetStatus();
  void fillDummyStatus();
  void fillStatus(DQMStore& dqm_store,
                  const SiStripDetCabling* cabling,
                  const TkDetMap* tkDetMap,
                  const TrackerTopology* tTopo);
  void fillStatusAtLumi(DQMStore& dqm_store);
  void printStatusReport();
  void fillFaultyModuleStatus(DQMStore& dqm_store, const TrackerTopology* tTopo);

private:
  struct SubDetMEs {
    MonitorElement* DetFraction;
    MonitorElement* SToNFlag;
    MonitorElement* SummaryFlag;
    std::string detectorTag;
  };

  void fillDetectorStatus(DQMStore& dqm_store, const SiStripDetCabling* cabling);
  void fillSubDetStatus(
      DQMStore& dqm_store, const SiStripDetCabling* cabling, SubDetMEs& mes, unsigned int xbin, float& gflag);
  void getModuleStatus(DQMStore& dqm_store,
                       std::vector<MonitorElement*>& layer_mes,
                       int& errdet,
                       int& errdet_hasBadChan,
                       int& errdet_hasTooManyDigis,
                       int& errdet_hasTooManyClu,
                       int& errdet_hasExclFed,
                       int& errdet_hasDcsErr);

  void fillStatusHistogram(MonitorElement const*, int xbin, int ybin, float val);
  void initialiseBadModuleList();

  void fillDetectorStatusAtLumi(DQMStore& dqm_store);

  std::map<std::string, SubDetMEs> SubDetMEsMap;
  std::map<std::string, std::string> SubDetFolderMap;

  MonitorElement* DetFractionReportMap{nullptr};
  MonitorElement* DetFractionReportMap_hasBadChan{nullptr};
  MonitorElement* DetFractionReportMap_hasTooManyDigis{nullptr};
  MonitorElement* DetFractionReportMap_hasTooManyClu{nullptr};
  MonitorElement* DetFractionReportMap_hasExclFed{nullptr};
  MonitorElement* DetFractionReportMap_hasDcsErr{nullptr};
  MonitorElement* SToNReportMap{nullptr};
  MonitorElement* SummaryReportMap{nullptr};
  MonitorElement* SummaryReportGlobal{nullptr};
  MonitorElement* TrackSummaryReportMap{nullptr};
  MonitorElement* TrackSummaryReportGlobal{nullptr};

  std::map<uint32_t, uint16_t> badModuleList;

  edm::ParameterSet const pSet_;

  bool bookedStripStatus_{false};

  const TkDetMap* tkDetMap_;
};
#endif
