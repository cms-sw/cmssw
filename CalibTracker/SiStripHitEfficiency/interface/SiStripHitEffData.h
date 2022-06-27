#ifndef CalibTracker_SiStripHitEfficiency_SiStripHitEffData_h
#define CalibTracker_SiStripHitEfficiency_SiStripHitEffData_h

#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <unordered_map>

struct SiStripHitEffData {
public:
  SiStripHitEffData() : EventStats(), FEDErrorOccupancy(nullptr) {}

  void fillTkMapFromMap() {
    for (const auto& [id, count] : fedErrorCounts) {
      FEDErrorOccupancy.fill(id, count);
    }
  }

  dqm::reco::MonitorElement* EventStats;
  std::unordered_map<uint32_t, int> fedErrorCounts;
  TkHistoMap FEDErrorOccupancy;
};

#endif
