#ifndef CalibTracker_SiStripHitEfficiency_SiStripHitEffData_h
#define CalibTracker_SiStripHitEfficiency_SiStripHitEffData_h

#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <unordered_map>

struct SiStripHitEffData {
public:
  SiStripHitEffData() : EventStats(), FEDErrorOccupancy(nullptr), m_isLoaded(false) {}

  void fillTkMapFromMap() {
    for (const auto& [id, count] : fedErrorCounts) {
      FEDErrorOccupancy->fill(id, count);
    }
  }

  void fillMapFromTkMap(const int nevents, const float threshold, const std::vector<DetId>& stripDetIds) {
    const auto& Maps = FEDErrorOccupancy->getAllMaps();
    std::vector<bool> isThere;
    isThere.reserve(Maps.size());
    std::transform(Maps.begin() + 1, Maps.end(), std::back_inserter(isThere), [](auto& x) { return !(x == nullptr); });

    int count{0};
    for (const auto& it : isThere) {
      count++;
      LogTrace("SiStripHitEffData") << " layer: " << count << " " << it << std::endl;
      if (it)
        LogTrace("SiStripHitEffData") << "resolving to " << Maps[count]->getName()
                                      << " with entries: " << Maps[count]->getEntries() << std::endl;
      // color the map
      Maps[count]->setOption("colz");
    }

    for (const auto& det : stripDetIds) {
      const auto& counts = FEDErrorOccupancy->getValue(det);

      if (counts > 0) {
        float fraction = counts / nevents;

        LogTrace("SiStripHitEffData") << det.rawId() << " has " << counts << " counts, " << fraction * 100
                                      << "% fraction of the " << nevents << " events processed" << std::endl;

        if (fraction > threshold) {
          fedErrorCounts.insert(std::make_pair(det, 1));
        }
      }  // do not check functioning modules
    }
    // the map has been loaded
    m_isLoaded = true;
  }

  const bool checkFedError(const DetId det) {
    if (m_isLoaded) {
      return (fedErrorCounts.find(det) == fedErrorCounts.end());
    } else {
      throw cms::Exception("LogicError") << __PRETTY_FUNCTION__ << "cannot check DetId when map not filled";
    }
  }

  dqm::reco::MonitorElement* EventStats;
  std::unordered_map<uint32_t, int> fedErrorCounts;
  std::unique_ptr<TkHistoMap> FEDErrorOccupancy;

private:
  bool m_isLoaded;
};

#endif
