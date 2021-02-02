#ifndef CUDADataFormats_SiPixelCluster_interface_gpuClusteringConstants_h
#define CUDADataFormats_SiPixelCluster_interface_gpuClusteringConstants_h

#include <cstdint>
#include <limits>

namespace pixelGPUConstants {
#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumberOfHits = 24 * 1024;
#else
  // data at pileup 50 has 18300 +/- 3500 hits; 40000 is around 6 sigma away
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumberOfHits = 48 * 1024;
#endif
}  // namespace pixelGPUConstants

namespace gpuClustering {
#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxHitsInIter() { return 64; }
#else
  // optimized for real data PU 50
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxHitsInIter() { return 160; }
#endif
  constexpr uint32_t maxHitsInModule() { return 1024; }

  constexpr uint16_t maxNumModules = 2000;
  constexpr int32_t maxNumClustersPerModules = maxHitsInModule();
  constexpr uint32_t maxNumClusters = pixelGPUConstants::maxNumberOfHits;
  constexpr uint16_t invalidModuleId = std::numeric_limits<uint16_t>::max() - 1;
  static_assert(invalidModuleId > maxNumModules);  // invalidModuleId must be > maxNumModules

}  // namespace gpuClustering

#endif  // CUDADataFormats_SiPixelCluster_interface_gpuClusteringConstants_h
