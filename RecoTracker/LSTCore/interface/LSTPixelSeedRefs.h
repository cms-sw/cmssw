#ifndef RecoTracker_LSTCore_interface_LSTPixelSeedRefs_h
#define RecoTracker_LSTCore_interface_LSTPixelSeedRefs_h

#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

namespace lst {

  // Host-only references to the input pixel seeds, in the order of the pixel seeds of LSTInput; read by
  // LSTOutputConverter.
  struct LSTPixelSeedRefs {
    std::vector<edm::Ref<TrajectorySeedCollection>> seeds;
  };

}  // namespace lst

#endif
