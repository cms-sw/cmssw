#ifndef DataFormats_SiPixelClusterSoA_interface_ClusteringConstants_h
#define DataFormats_SiPixelClusterSoA_interface_ClusteringConstants_h

#include <cstdint>
#include <limits>

//TODO: move this to TrackerTraits!
namespace pixelClustering {
#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxHitsInIter() { return 64; }
#else
  // optimized for real data PU 50
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxHitsInIter() { return 160; }  //TODO better tuning for PU 140-200
#endif
  constexpr uint32_t maxHitsInModule() { return 1024; }

  constexpr uint16_t clusterThresholdLayerOne = 2000;
  constexpr uint16_t clusterThresholdOtherLayers = 4000;

  constexpr uint16_t clusterThresholdPhase2LayerOne = 4000;
  constexpr uint16_t clusterThresholdPhase2OtherLayers = 4000;

  constexpr uint32_t maxNumDigis = 3 * 256 * 1024;  // @PU=200 µ=530k σ=50k this is >4σ away
  constexpr uint16_t maxNumModules = 4000;

  constexpr int32_t maxNumClustersPerModules = maxHitsInModule();
  constexpr uint16_t invalidModuleId = std::numeric_limits<uint16_t>::max() - 1;
  constexpr int invalidClusterId = -9999;
  static_assert(invalidModuleId > maxNumModules);  // invalidModuleId must be > maxNumModules

}  // namespace pixelClustering

#endif  // DataFormats_SiPixelClusterSoA_interface_ClusteringConstants_h
