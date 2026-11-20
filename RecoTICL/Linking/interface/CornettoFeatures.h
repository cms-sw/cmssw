#ifndef RecoTICL_Linking_CornettoFeatures_h
#define RecoTICL_Linking_CornettoFeatures_h

// Per-trackster and per-group quantities of the Cornetto pair test and emission rule.

#include <vector>

#include "CLHEP/Units/PhysicalConstants.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/Vector3D.h"

namespace ticl::cornetto {

  inline constexpr float kInvSpeedOfLight = 1.f / static_cast<float>(CLHEP::c_light * CLHEP::ns / CLHEP::cm);  // ns/cm

  // Number of ticl::Trackster::ParticleType slots.
  inline constexpr int kNumParticleTypes = static_cast<int>(Trackster::ParticleType::unknown) + 1;

  // The EM probability of a trackster: the sum of its id_probabilities over the emCategories (ParticleType slots)
  // divided by the sum over all slots. -1 without a PID: no probability, or only unknown (a merged trackster).
  float emProbability(const Trackster& ts, const std::vector<int>& emCategories);

  // Quantities the pair test reads, one entry per trackster or per group.
  struct Features {
    std::vector<math::XYZVectorF> bary;
    std::vector<float> energy;
    std::vector<float> eta;
    std::vector<float> phi;
    std::vector<float> rad;   // |bary|
    std::vector<float> em;    // EM probability, negative without a PID
    std::vector<float> time;  // local to the barycenter [ns]
    std::vector<float> timeError;
    std::vector<bool> valid;  // finite barycenter and energy

    void resize(std::size_t n);
  };

  // Features of the groups from the features f of their valid members: the raw energy, the energy-weighted
  // barycenter, the energy-weighted EM probability of the members with a PID (-1 without one), and the error-weighted
  // time of the members with a positive time error moved to the group barycenter (time error -1 without one). A group
  // is valid when its energy is positive and finite and its barycenter is finite.
  Features groupFeatures(const Features& f, const std::vector<std::vector<unsigned int>>& groups);

}  // namespace ticl::cornetto

#endif
