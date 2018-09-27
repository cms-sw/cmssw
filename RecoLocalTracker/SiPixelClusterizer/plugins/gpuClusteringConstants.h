#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusteringConstants_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusteringConstants_h

#include <cstdint>

namespace gpuClustering {
  constexpr uint32_t MaxNumModules  = 2000;
  constexpr uint32_t MaxNumPixels   = 256 * 2000;   // this does not mean maxPixelPerModule == 256!
  constexpr uint32_t MaxNumClustersPerModules = 1024;
  constexpr uint16_t InvId          = 9999;         // must be > MaxNumModules

}

#endif // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusteringConstants_h
