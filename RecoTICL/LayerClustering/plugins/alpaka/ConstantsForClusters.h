#ifndef RecoLocalCalo_HGCalRecProducers_interface_alpaka_ConstantsForClusters_h
#define RecoLocalCalo_HGCalRecProducers_interface_alpaka_ConstantsForClusters_h

#include <cstdint>

namespace hgcal::constants {
  static constexpr int kHGCalLayers = 96;
  static constexpr int kInvalidCluster = -1;
  static constexpr uint8_t kInvalidClusterByte = 0xff;
}  // namespace hgcal::constants

#endif  // RecoLocalCalo_HGCalRecProducers_interface_alpaka_ConstantsForClusters_h
