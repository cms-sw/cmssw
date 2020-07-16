#ifndef L1Trigger_TrackFindingTMTT_PrintL1trk
#define L1Trigger_TrackFindingTMTT_PrintL1trk

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>

// Use LogVerbatim with "L1track" category & floating point precision specified here.
// Example use: PrintL1trk() << "My message "<<x<<" more text".

namespace tmtt {

  class PrintL1trk {
  public:
    PrintL1trk(unsigned int nDigits = 4) : lv_("L1track"), nDigits_(nDigits){};

    template <class T>
    edm::LogVerbatim& operator<<(const T& t) {
      lv_ << std::fixed << std::setprecision(nDigits_) << t;
      return lv_;
    }

  private:
    edm::LogVerbatim lv_;
    const unsigned int nDigits_;
  };

}  // namespace tmtt

#endif
