// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"

#include <algorithm>

#include "HepPDT/ParticleID.hh"

namespace truth {

  bool BranchSelector::operator()(Branch const& branch) const {
    return passesNonKinematic(branch) && failedKinematicCuts(branch) == 0u;
  }

  bool BranchSelector::passesNonKinematic(Branch const& branch) const {
    if (!branch.valid())
      return false;

    if (config_.signalOnly && !branch.isSignal())
      return false;

    if (config_.intimeOnly && !branch.isInTime())
      return false;

    const int32_t pdgId = branch.rootPdgId();

    if (config_.chargedOnly && HepPDT::ParticleID(pdgId).threeCharge() == 0)
      return false;

    if (!config_.pdgIds.empty() &&
        std::find(config_.pdgIds.begin(), config_.pdgIds.end(), pdgId) == config_.pdgIds.end())
      return false;

    return true;
  }

  uint32_t BranchSelector::failedKinematicCuts(Branch const& branch) const {
    if (!branch.valid())
      return 0u;

    // Kinematics from the defining root particle. Copy by value: root() returns
    // a temporary Particle, so a reference to its momentum() would dangle.
    const auto rootParticle = branch.root();

    // Skip the kinematic cuts for a root Geant4 never tracked. That is the line between
    // an object whose momentum a detector could measure and one whose momentum is a
    // bookkeeping quantity: a resonance is GEN-only, decays before anything, and at rest
    // carries pt about 0 with |eta| unbounded, so a track-shaped cut throws it away while
    // its decay products fill the calorimeter. A pion that showers has a SimTrack and its
    // pt IS an observable, so it stays subject to the cuts even though it also "decayed".
    if (config_.kinematicsOnStableOnly && !rootParticle.data().hasSim())
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
