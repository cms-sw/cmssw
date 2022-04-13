#ifndef SiStripHitEfficiencyHelpers_H
#define SiStripHitEfficiencyHelpers_H

// A bunch of helper functions to deal with menial tasks in the
// hit efficiency computation for the PCL workflow

#include <string>
#include <fmt/printf.h>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace {

  inline unsigned int checkLayer(unsigned int iidd, const TrackerTopology* tTopo) {
    switch (DetId(iidd).subdetId()) {
      case SiStripSubdetector::TIB:
        return tTopo->tibLayer(iidd);
      case SiStripSubdetector::TOB:
        return tTopo->tobLayer(iidd) + 4;
      case SiStripSubdetector::TID:
        return tTopo->tidWheel(iidd) + 10;
      case SiStripSubdetector::TEC:
        return tTopo->tecWheel(iidd) + 13;
      default:
        return 0;
    }
  }

  inline std::string layerName(unsigned int k, const bool showRings, const unsigned int nTEClayers) {
    const std::string ringlabel{showRings ? "R" : "D"};
    if (k > 0 && k < 5) {
      return fmt::format("TIB L{:d}", k);
    } else if (k > 4 && k < 11) {
      return fmt::format("TOB L{:d}", k - 4);
    } else if (k > 10 && k < 14) {
      return fmt::format("TID {0}{1:d}", ringlabel, k - 10);
    } else if (k > 13 && k < 14 + nTEClayers) {
      return fmt::format("TEC {0}{1:d}", ringlabel, k - 13);
    } else {
      return "";
    }
  }

}  // namespace
#endif
