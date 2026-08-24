// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_interface_BranchSelector_h
#define PhysicsTools_TruthInfo_interface_BranchSelector_h

#include <cstdint>
#include <limits>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/Branch.h"

namespace truth {

  // Kinematic / provenance selection of truth Branches, mirroring the cut
  // surface of TrackingParticleSelector / CaloParticleSelector but applied to a
  // Branch (the dynamic successor of TrackingParticle/CaloParticle). The branch
  // kinematics are taken from its defining root particle.
  class BranchSelector {
  public:
    struct Config {
      // Unbounded by default: the sentinels are the type's own limits, so "no cut"
      // needs no magic constant and survives a change of representation.
      float ptMin = 0.f;
      float ptMax = std::numeric_limits<float>::max();
      float etaMin = std::numeric_limits<float>::lowest();
      float etaMax = std::numeric_limits<float>::max();
      std::vector<int32_t> pdgIds;  // empty = accept all; matched on signed PDG id
      bool signalOnly = false;      // bunchCrossing == 0 and event == 0
      bool intimeOnly = false;      // bunchCrossing == 0
      bool chargedOnly = false;     // root particle electrically charged
      bool invertEta = false;       // keep |eta| OUTSIDE [etaMin, etaMax]
      // Apply the pt and eta cuts ONLY to a root Geant4 actually tracked. The cuts describe what a detector can see, and the momentum of a
      // root that decayed is not a detector observable: a resonance produced at rest
      // carries pt about 0 and therefore |eta| -> infinity, so a pt>1 GeV or |eta|<4 cut
      // throws it away while its decay products are all over the calorimeter. Measured on
      // DYToLL: 43% of the Z bosons were rejected that way, every one of them with a
      // subgraph full of sim-hits.
      bool kinematicsOnStableOnly = true;
    };

    // The cuts that are ALSO efficiency-plot axes. An efficiency drawn against pt must not
    // have the pt cut applied to its own denominator, or the cut deforms the turn-on it is
    // meant to show: measured on no-PU ttbar, the caloBoundary denominator in the first pt
    // bin is 10024 with the cut and 144529 without, a factor 14.4, while the second bin
    // moves by 1.05. So these two are reported per branch instead of being folded into a
    // single accept/reject, and the consumer decides per axis.
    //
    // Only these two: every other cut here is a provenance or species requirement that no
    // plot uses as an x axis, so suppressing one would answer no question.
    enum class CutBit : uint32_t { None = 0, Pt = 1u << 0, Eta = 1u << 1 };

    BranchSelector() = default;
    explicit BranchSelector(Config config) : config_(std::move(config)) {}

    // Unchanged meaning: passes everything. Equivalent to passesNonKinematic() with no
    // kinematic cut failed.
    [[nodiscard]] bool operator()(Branch const& branch) const;

    // The cuts that are not plot axes. A branch failing any of them is not a candidate at
    // all and may not enter any plot, whatever its kinematics.
    [[nodiscard]] bool passesNonKinematic(Branch const& branch) const;

    // Which of the plotted-axis cuts this branch FAILS; 0 means it passes both. A branch
    // whose root Geant4 never tracked fails neither, since the kinematic cuts do not
    // apply to it at all.
    [[nodiscard]] uint32_t failedKinematicCuts(Branch const& branch) const;

    [[nodiscard]] Config const& config() const { return config_; }

  private:
    Config config_;
  };

}  // namespace truth

#endif
