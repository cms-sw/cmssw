#ifndef RecoTracker_LSTCore_interface_LSTOTHits_h
#define RecoTracker_LSTCore_interface_LSTOTHits_h

#include <vector>

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

namespace lst {

  // Host-only pointers to the OT rechits, in the order of the OT hits of LSTInput; read by LSTOutputConverter.
  // Wrapped in a struct because a std::vector<T*> event product fails the framework's contained-type dictionary check.
  struct LSTOTHits {
    std::vector<TrackingRecHit const*> hits;
  };

}  // namespace lst

#endif
