#ifndef L1Trigger_Phase2L1GT_L1GTOfflineThreshold_h
#define L1Trigger_Phase2L1GT_L1GTOfflineThreshold_h

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace l1t {

  // Rounds to one decimal digit exactly like Python's round(value, 1):
  // the exact binary value is rounded (ties to even) and the closest double
  // to the resulting decimal is returned.
  inline double roundToOneDecimal(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.1f", value);
    return std::strtod(buffer, nullptr);
  }

  // Converts an offline threshold into the online threshold using the linear
  // object scaling offline = slope * online + offset, i.e.
  //   online = max(minOnline, round((offline - offset) / slope, 1))
  // This mirrors off2onl_thresholds() of the menu python helpers, so the
  // emulator, flattened menus and the firmware writer use identical values.
  inline double offlineToOnlinePt(double offline, double offset, double slope, double minOnline = 0.) {
    return std::max(minOnline, roundToOneDecimal((offline - offset) / slope));
  }

}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTOfflineThreshold_h
