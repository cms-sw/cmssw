#ifndef RecoTICL_Inference_TracksterLayerClusterProfile_h
#define RecoTICL_Inference_TracksterLayerClusterProfile_h

#include <array>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

namespace ticl {

  // Radii (cm) around the main axis of the two core energy shares of the profile.
  inline constexpr float kProfileCore2Radius = 2.f;
  inline constexpr float kProfileCore5Radius = 5.f;

  // Layer-cluster profile of a trackster around its main axis: the largest layer-cluster energy share, the energy
  // shares within 2 and 5 cm of the axis, the energy-weighted transverse rms (cm), the smallest |z| and the |z| extent
  // (cm). A zero main axis is replaced by the direction of the barycenter. Without layer-cluster energy the first five
  // are -1 and the extent is 0; a non-finite value is 0.
  std::array<float, 6> layerClusterProfile(const Trackster& trackster,
                                           const std::vector<reco::CaloCluster>& layerClusters);

}  // namespace ticl

#endif
