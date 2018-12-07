#ifndef RecoLocalTracker_SiPixelClusterizer_interface_PixelTrackingGPUConstants_h
#define RecoLocalTracker_SiPixelClusterizer_interface_PixelTrackingGPUConstants_h

#include <cstdint>

namespace PixelGPUConstants {
  constexpr uint16_t maxNumberOfHits = 40000;       // data at pileup 50 has 18300 +/- 3500 hits; 40000 is around 6 sigma away

}

#endif // RecoLocalTracker_SiPixelClusterizer_interface_PixelTrackingGPUConstants_h
