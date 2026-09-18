#ifndef RecoLocalCalo_HGCalRecProducers_interface_alpaka_ConstantsForClusters_h
#define RecoLocalCalo_HGCalRecProducers_interface_alpaka_ConstantsForClusters_h

namespace hgcal::constants {
  static constexpr int kHGCalLayers = 96;
  // Physical layer count: HGCAL has 47 layers per endcap. The rechit SoA encodes
  // the layer as a single global index (layerOnSide + zside * lastLayer), so both
  // endcaps together span 2 * 47 = 94 distinct layer slots.
  static constexpr int kHGCalLayersPerEndcap = 47;
  static constexpr int kHGCalLayersBothEndcaps = 2 * kHGCalLayersPerEndcap;
  static constexpr int kInvalidCluster = -1;
  static constexpr uint8_t kInvalidClusterByte = 0xff;
  static constexpr int kInvalidIndex = -1;
  static constexpr uint8_t kInvalidIndexByte = 0xff;
  // The nearestHigher column is unsigned, so "no nearest higher" (a seed or an
  // outlier) cannot be the -1 the algorithms use internally.
  static constexpr unsigned int kInvalidNearestHigher = 0xffffffffu;
}  // namespace hgcal::constants

#endif  // RecoLocalCalo_HGCalRecProducers_interface_alpaka_ConstantsForClusters_h
