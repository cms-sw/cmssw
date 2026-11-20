#include "RecoTICL/Inference/interface/TracksterLayerClusterProfile.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ticl {

  std::array<float, 6> layerClusterProfile(const Trackster& trackster,
                                           const std::vector<reco::CaloCluster>& layerClusters) {
    const auto& b = trackster.barycenter();
    auto axis = trackster.eigenvectors(0);
    if (!(axis.mag2() > 0.f) || !std::isfinite(axis.mag2()))
      axis = b;
    axis = axis.unit();
    float eSum = 0.f, eMax = 0.f, eCore2 = 0.f, eCore5 = 0.f, r2Sum = 0.f;
    float zMin = std::numeric_limits<float>::max(), zMax = 0.f;
    for (size_t i = 0; i < trackster.vertices().size(); ++i) {
      const auto& lc = layerClusters[trackster.vertices(i)];
      const float e = lc.energy() / std::max<float>(1.f, trackster.vertex_multiplicity(i));
      const Trackster::Vector d(lc.x() - b.x(), lc.y() - b.y(), lc.z() - b.z());
      const float along = d.Dot(axis);
      const float r = std::sqrt(std::max(0.f, d.mag2() - along * along));
      eSum += e;
      eMax = std::max(eMax, e);
      eCore2 += r < kProfileCore2Radius ? e : 0.f;
      eCore5 += r < kProfileCore5Radius ? e : 0.f;
      r2Sum += e * r * r;
      zMin = std::min(zMin, std::abs(static_cast<float>(lc.z())));
      zMax = std::max(zMax, std::abs(static_cast<float>(lc.z())));
    }
    std::array<float, 6> profile{-1.f, -1.f, -1.f, -1.f, -1.f, 0.f};
    if (eSum > 0.f)
      profile = {eMax / eSum, eCore2 / eSum, eCore5 / eSum, std::sqrt(r2Sum / eSum), zMin, zMax - zMin};
    for (auto& v : profile)
      if (!std::isfinite(v))
        v = 0.f;
    return profile;
  }

}  // namespace ticl
