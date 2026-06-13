#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"

#include <algorithm>

#include "HepPDT/ParticleID.hh"

namespace truth {

  bool BranchSelector::operator()(Branch const& branch) const {
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

    // Kinematics from the defining root particle.
    auto const& p4 = branch.root().momentum();
    const double pt = p4.pt();
    if (pt < config_.ptMin || pt > config_.ptMax)
      return false;

    const double eta = p4.eta();
    const bool insideEta = eta >= config_.etaMin && eta <= config_.etaMax;
    if (config_.invertEta ? insideEta : !insideEta)
      return false;

    return true;
  }

}  // namespace truth
