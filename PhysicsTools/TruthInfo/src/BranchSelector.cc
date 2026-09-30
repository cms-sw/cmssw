// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"

#include <algorithm>
#include <cstdlib>

#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"

namespace truth {

  namespace {
    // Whether a detector measures the momentum of this particle, directly or as the sum
    // of its decay products.
    bool hasObservableMomentum(ParticleData const& particle) {
      if (particle.hasSim() || particle.status == 1)
        return true;
      if (isShowerObject(particle.pdgId))
        return false;
      const int32_t a = std::abs(particle.pdgId);
      const bool photonOrLepton = a == 22 || (a >= 11 && a <= 16);
      const bool hadron = a >= 100 && a < 1000000;
      return photonOrLepton || hadron;
    }
  }  // namespace

  bool BranchSelector::operator()(Branch const& branch) const {
    return passesNonKinematic(branch) && failedKinematicCuts(branch) == 0u;
  }

  bool BranchSelector::passesNonKinematic(Branch const& branch) const {
    if (config_.signalOnly && !branch.isSignal())
      return false;

    if (config_.intimeOnly && !branch.isInTime())
      return false;

    if (config_.chargedOnly && branch.root().charge() == 0.)
      return false;

    const int32_t pdgId = branch.rootPdgId();
    if (!config_.pdgIds.empty() &&
        std::find(config_.pdgIds.begin(), config_.pdgIds.end(), pdgId) == config_.pdgIds.end())
      return false;

    return true;
  }

  uint32_t BranchSelector::failedKinematicCuts(Branch const& branch) const {
    // Kinematics from the defining root particle. Copy by value: root() returns
    // a temporary Particle, so a reference to its momentum() would dangle.
    const auto rootParticle = branch.root();

    // A resonance at rest carries pt about 0 with |eta| unbounded, so a track-shaped cut
    // would throw it away while its decay products fill the calorimeter.
    if (config_.kinematicsOnStableOnly && !hasObservableMomentum(rootParticle.data()))
      return 0u;

    uint32_t failed = 0u;

    const auto& p4 = rootParticle.momentum();
    const double pt = p4.pt();
    if (pt < config_.ptMin || pt > config_.ptMax)
      failed |= static_cast<uint32_t>(CutBit::Pt);

    const double eta = p4.eta();
    const bool insideEta = eta >= config_.etaMin && eta <= config_.etaMax;
    if (config_.invertEta ? insideEta : !insideEta)
      failed |= static_cast<uint32_t>(CutBit::Eta);

    return failed;
  }

}  // namespace truth
