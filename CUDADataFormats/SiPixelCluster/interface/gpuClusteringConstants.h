#ifndef CUDADataFormats_SiPixelCluster_interface_gpuClusteringConstants_h
#define CUDADataFormats_SiPixelCluster_interface_gpuClusteringConstants_h

#include <cstdint>
#include <limits>

namespace gpuClustering {
#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxHitsInIter() { return 64; }
#else
  // optimized for real data PU 50
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxHitsInIter() { return 160; }  //TODO better tuning for PU 140-200
#endif

  constexpr uint16_t clusterThresholdLayerOne = 2000;
  constexpr uint16_t clusterThresholdOtherLayers = 4000;

  constexpr uint32_t maxNumDigis = 3 * 256 * 1024;  // @PU=200 µ=530 σ=50k this is >4σ away
  constexpr uint16_t maxNumModules = 4000;

  constexpr uint16_t invalidModuleId = std::numeric_limits<uint16_t>::max() - 1;
  constexpr int invalidClusterId = -9999;
  static_assert(invalidModuleId > maxNumModules);  // invalidModuleId must be > maxNumModules

}  // namespace gpuClustering

#endif  // CUDADataFormats_SiPixelCluster_interface_gpuClusteringConstants_h
