#ifndef RecoLocalTracker_SiStripClusterizer_plugins_StripDataView_h
#define RecoLocalTracker_SiStripClusterizer_plugins_StripDataView_h

#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include <cstdint>

struct ChannelLocsView;

namespace stripgpu {
  static constexpr auto kMaxSeedStrips = 200000;
  static constexpr uint32_t kClusterMaxStrips = SiStripClustersCUDADevice::kClusterMaxStrips;

  struct StripDataView {
    const ChannelLocsView *chanlocs;
    uint8_t *adc;
    uint16_t *channel;
    stripgpu::stripId_t *stripId;
    int *seedStripsNCIndex, *seedStripsMask, *seedStripsNCMask, *prefixSeedStripsNCMask;
    int nSeedStripsNC;
    int nStrips;
    float channelThreshold, seedThreshold, clusterThresholdSquared;
    uint8_t maxSequentialHoles, maxSequentialBad, maxAdjacentBad;
    float minGoodCharge;
    int clusterSizeLimit;
  };
}  // namespace stripgpu
#endif
