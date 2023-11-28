#ifndef RecoLocalTracker_SiStripClusterizer_plugins_StripDataView_h
#define RecoLocalTracker_SiStripClusterizer_plugins_StripDataView_h

#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"
#include "FWCore/Utilities/interface/HostDeviceConstant.h"

#include <cstdint>

class ChannelLocsView;

namespace stripgpu {
  HOST_DEVICE_CONSTANT auto kMaxSeedStrips = 200000;

  struct StripDataView {
    const ChannelLocsView *chanlocs;
    uint8_t *adc;
    uint16_t *channel;
    stripId_t *stripId;
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
