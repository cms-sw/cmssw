#include <cmath>

#include "RecoTICL/Linking/interface/CornettoFeatures.h"

namespace ticl::cornetto {

  float emProbability(const Trackster& ts, const std::vector<int>& emCategories) {
    float sum = 0.f;
    for (float v : ts.id_probabilities())
      sum += v;
    if (!(sum > ts.id_probability(Trackster::ParticleType::unknown)))
      return -1.f;
    float em = 0.f;
    for (int category : emCategories)
      em += ts.id_probabilities(category);
    return em / sum;
  }

  void Features::resize(std::size_t n) {
    bary.resize(n);
    energy.resize(n);
    eta.resize(n);
    phi.resize(n);
    rad.resize(n);
    em.resize(n);
    time.resize(n);
    timeError.resize(n);
    valid.resize(n);
  }

  Features groupFeatures(const Features& f, const std::vector<std::vector<unsigned int>>& groups) {
    const std::size_t n = groups.size();
    Features g;
    g.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
      float energy = 0.f;
      float emWeight = 0.f;
      float em = 0.f;
      math::XYZVectorF weighted;
      for (unsigned int i : groups[k]) {
        if (!f.valid[i])
          continue;
        energy += f.energy[i];
        weighted += f.bary[i] * f.energy[i];
        if (f.em[i] >= 0.f) {
          emWeight += f.energy[i];
          em += f.em[i] * f.energy[i];
        }
      }
      const auto bary = energy > 0.f ? weighted / energy : weighted;
      const float rad = std::sqrt(bary.Mag2());
      // Member times moved to the group barycenter along the direction from the origin, weighted by their errors.
      float sumW = 0.f;
      float sumWT = 0.f;
      for (unsigned int i : groups[k]) {
        if (!f.valid[i] || !(f.timeError[i] > 0.f))
          continue;
        const float w = 1.f / (f.timeError[i] * f.timeError[i]);
        sumW += w;
        sumWT += w * (f.time[i] + (rad - f.rad[i]) * kInvSpeedOfLight);
      }
      g.bary[k] = bary;
      g.energy[k] = energy;
      g.eta[k] = bary.eta();
      g.phi[k] = bary.phi();
      g.rad[k] = rad;
      g.em[k] = emWeight > 0.f ? em / emWeight : -1.f;
      g.time[k] = sumW > 0.f ? sumWT / sumW : 0.f;
      g.timeError[k] = sumW > 0.f ? 1.f / std::sqrt(sumW) : -1.f;
      g.valid[k] = energy > 0.f && std::isfinite(bary.x()) && std::isfinite(bary.y()) && std::isfinite(bary.z()) &&
                   std::isfinite(energy);
    }
    return g;
  }

}  // namespace ticl::cornetto
