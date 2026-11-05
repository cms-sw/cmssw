#ifndef DataFormats_HGCalReco_InterpretationHypotheses_h
#define DataFormats_HGCalReco_InterpretationHypotheses_h

#include <cassert>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include "DataFormats/Math/interface/Vector3D.h"

namespace ticl {
  namespace io_v1 {
    // The competing hypotheses of the TICLv6 interpretation stage, their conflicts and the solver settings.
    class InterpretationHypotheses {
    public:
      // The values are persisted: a new type goes at the end.
      enum class Type : std::uint8_t {
        Muon,
        Electron,
        Photon,
        ChargedHadron,
        NeutralHadron,
        Jet,
        RecoveryChargedHadron
      };
      using Vector = math::XYZVectorF;

      // One hypothesis. tracks are indices into the track collection of the interpretation stage; a footprint of zero
      // raw energy means no footprint.
      struct Entry {
        Type type;
        std::span<const int> tracks;
        float weight;
        bool accepted;
        float rawEnergy;
        float rawEmEnergy;
        Vector barycenter;
      };

      InterpretationHypotheses() = default;

      // Adds a hypothesis and returns its index.
      unsigned int add(const Entry &e) {
        type_.push_back(static_cast<std::uint8_t>(e.type));
        if (trackOffsets_.empty())
          trackOffsets_.push_back(0);
        const int *first = e.tracks.data();
        if (std::greater_equal<>()(first, tracks_.data()) && std::less<>()(first, tracks_.data() + tracks_.size())) {
          // The tracks are a part of tracks_: copy them first, because the insertion can reallocate tracks_.
          const std::vector<int> copy(e.tracks.begin(), e.tracks.end());
          tracks_.insert(tracks_.end(), copy.begin(), copy.end());
        } else {
          tracks_.insert(tracks_.end(), e.tracks.begin(), e.tracks.end());
        }
        trackOffsets_.push_back(static_cast<std::uint32_t>(tracks_.size()));
        weight_.push_back(e.weight);
        accepted_.push_back(e.accepted ? 1 : 0);
        rawEnergy_.push_back(e.rawEnergy);
        rawEmEnergy_.push_back(e.rawEmEnergy);
        x_.push_back(e.barycenter.x());
        y_.push_back(e.barycenter.y());
        z_.push_back(e.barycenter.z());
        return static_cast<unsigned int>(type_.size() - 1);
      }
      // Adds the conflict between hypotheses i and j, i < j.
      void addConflict(unsigned int i, unsigned int j) {
        assert(i < j && j < size());
        conflictFirst_.push_back(i);
        conflictSecond_.push_back(j);
      }
      void setTrackBonus(float bonus) { trackBonus_ = bonus; }
      void setSolverLimits(unsigned int maxExactComponent, std::uint64_t searchBudget) {
        maxExactComponent_ = maxExactComponent;
        searchBudget_ = searchBudget;
      }

      unsigned int size() const { return static_cast<unsigned int>(type_.size()); }
      Type type(unsigned int i) const { return static_cast<Type>(type_[i]); }
      std::span<const int> tracks(unsigned int i) const {
        return {tracks_.data() + trackOffsets_[i], tracks_.data() + trackOffsets_[i + 1]};
      }
      // Calibrated logit of the hypothesis model; the fixed weight of a track-only hypothesis; -infinity: vetoed.
      float weight(unsigned int i) const { return weight_[i]; }
      // The weight that the solver compares: weight plus trackBonus per track.
      float solverWeight(unsigned int i) const {
        return weight_[i] + trackBonus_ * static_cast<float>(trackOffsets_[i + 1] - trackOffsets_[i]);
      }
      float trackBonus() const { return trackBonus_; }
      unsigned int maxExactComponent() const { return maxExactComponent_; }
      std::uint64_t searchBudget() const { return searchBudget_; }
      bool accepted(unsigned int i) const { return accepted_[i] != 0; }
      float rawEnergy(unsigned int i) const { return rawEnergy_[i]; }
      float rawEmEnergy(unsigned int i) const { return rawEmEnergy_[i]; }
      Vector barycenter(unsigned int i) const { return Vector(x_[i], y_[i], z_[i]); }

      unsigned int nConflicts() const { return static_cast<unsigned int>(conflictFirst_.size()); }
      std::pair<unsigned int, unsigned int> conflict(unsigned int k) const {
        return {conflictFirst_[k], conflictSecond_[k]};
      }

    private:
      std::vector<std::uint8_t> type_;
      std::vector<std::uint32_t> trackOffsets_;
      std::vector<int> tracks_;
      std::vector<float> weight_;
      std::vector<std::uint8_t> accepted_;
      std::vector<float> rawEnergy_;
      std::vector<float> rawEmEnergy_;
      std::vector<float> x_;
      std::vector<float> y_;
      std::vector<float> z_;
      std::vector<std::uint32_t> conflictFirst_;
      std::vector<std::uint32_t> conflictSecond_;
      float trackBonus_ = 0.f;
      unsigned int maxExactComponent_ = 0;
      std::uint64_t searchBudget_ = 0;
    };
  }  // namespace io_v1
  using InterpretationHypotheses = io_v1::InterpretationHypotheses;
}  // namespace ticl
#endif
