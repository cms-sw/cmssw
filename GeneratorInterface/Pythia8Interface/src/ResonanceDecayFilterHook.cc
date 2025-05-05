#include "Pythia8/Pythia.h"
#include "GeneratorInterface/Pythia8Interface/interface/ResonanceDecayFilterHook.h"

using namespace Pythia8;

//--------------------------------------------------------------------------
bool ResonanceDecayFilterHook::initAfterBeams() {
  counter_event_ = 0;
  filter_ = settingsPtr->flag("ResonanceDecayFilter:filter");
  exclusive_ = settingsPtr->flag("ResonanceDecayFilter:exclusive");
  matching_ = settingsPtr->flag("ResonanceDecayFilter:matching");
  eMuAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:eMuAsEquivalent");
  eMuTauAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:eMuTauAsEquivalent");
  allNuAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:allNuAsEquivalent");
  udscAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:udscAsEquivalent");
  udscbAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:udscbAsEquivalent");
  wzAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:wzAsEquivalent");
  auto mothers = settingsPtr->mvec("ResonanceDecayFilter:mothers");
  mothers_.clear();
  mothers_.insert(mothers.begin(), mothers.end());
  daughters_ = settingsPtr->mvec("ResonanceDecayFilter:daughters");
  matchedDecays_ = settingsPtr->mvec("ResonanceDecayFilter:matchedDecays");

  requestedDaughters_.clear();

  for (int id : daughters_) {
    int did = std::abs(id);
    if (did == 13 && (eMuAsEquivalent_ || eMuTauAsEquivalent_)) {
      did = 11;
    }
    if (did == 15 && eMuTauAsEquivalent_) {
      did = 11;
    }
    if ((did == 14 || did == 16) && allNuAsEquivalent_) {
      did = 12;
    }
    if ((did == 2 || did == 3 || did == 4) && udscAsEquivalent_) {
      did = 1;
    }
    if ((did == 2 || did == 3 || did == 4 || did == 5) && udscbAsEquivalent_) {
      did = 1;
    }
    if ((did == 23 || did == 24) && wzAsEquivalent_) {
      did = 23;
    }

    ++requestedDaughters_[std::abs(did)];
  }

  requestedDecays_.clear();

  if (matching_) {
    if (matchedDecays_.empty() || matchedDecays_.size() % 3 != 0) {
      std::cerr << "Error: ResonanceDecayFilter:matchedDecays must be a multiple of 3 (mother, daughter1, daughter2)."
                << std::endl;
      return false;
    }

    for (size_t i = 0; i < matchedDecays_.size(); i += 3) {
      int mother = std::abs(matchedDecays_[i]);

      int d1 = std::abs(matchedDecays_[i + 1]);
      if (d1 == 13 && (eMuAsEquivalent_ || eMuTauAsEquivalent_)) {
        d1 = 11;
      }
      if (d1 == 15 && eMuTauAsEquivalent_) {
        d1 = 11;
      }
      if ((d1 == 14 || d1 == 16) && allNuAsEquivalent_) {
        d1 = 12;
      }
      if ((d1 == 2 || d1 == 3 || d1 == 4) && udscAsEquivalent_) {
        d1 = 1;
      }
      if ((d1 == 2 || d1 == 3 || d1 == 4 || d1 == 5) && udscbAsEquivalent_) {
        d1 = 1;
      }
      if ((d1 == 23 || d1 == 24) && wzAsEquivalent_) {
        d1 = 23;
      }
      requestedDecays_.insert({mother, d1});

      int d2 = std::abs(matchedDecays_[i + 2]);
      if (d2 == 13 && (eMuAsEquivalent_ || eMuTauAsEquivalent_)) {
        d2 = 11;
      }
      if (d2 == 15 && eMuTauAsEquivalent_) {
        d2 = 11;
      }
      if ((d2 == 14 || d2 == 16) && allNuAsEquivalent_) {
        d2 = 12;
      }
      if ((d2 == 2 || d2 == 3 || d2 == 4) && udscAsEquivalent_) {
        d2 = 1;
      }
      if ((d2 == 2 || d2 == 3 || d2 == 4 || d2 == 5) && udscbAsEquivalent_) {
        d2 = 1;
      }
      if ((d2 == 23 || d2 == 24) && wzAsEquivalent_) {
        d2 = 23;
      }
      requestedDecays_.insert({mother, d2});
    }
  }

  return true;
}

//--------------------------------------------------------------------------
bool ResonanceDecayFilterHook::checkVetoResonanceDecays(const Event &process) {
  if (!filter_)
    return false;

  //count the number of times hook is called.
  counter_event_++;

  observedDaughters_.clear();

  remainingDecays_.clear();
  if (matching_) {
    remainingDecays_ = requestedDecays_;
  }

  //count decay products
  for (int i = 0; i < process.size(); ++i) {
    const Particle &p = process[i];

    int did = std::abs(p.id());

    if (did == 13 && (eMuAsEquivalent_ || eMuTauAsEquivalent_)) {
      did = 11;
    }
    if (did == 15 && eMuTauAsEquivalent_) {
      did = 11;
    }
    if ((did == 14 || did == 16) && allNuAsEquivalent_) {
      did = 12;
    }
    if ((did == 2 || did == 3 || did == 4) && udscAsEquivalent_) {
      did = 1;
    }
    if ((did == 2 || did == 3 || did == 4 || did == 5) && udscbAsEquivalent_) {
      did = 1;
    }
    if ((did == 23 || did == 24) && wzAsEquivalent_) {
      did = 23;
    }

    int mid = p.mother1() > 0 ? std::abs(process[p.mother1()].id()) : 0;

    //if no list of mothers is provided, then all particles
    //in hard process and resonance decays are counted together
    if (mothers_.empty() || mothers_.count(mid) || mothers_.count(-mid)) {
      ++observedDaughters_[did];
    }

    if (matching_) {
      std::pair<int, int> decayPair = {mid, did};
      auto found = remainingDecays_.find(decayPair);
      if (found != remainingDecays_.end()) {
        remainingDecays_.erase(found);
      }
    }
  }

  //check if criteria is satisfied
  //inclusive mode: at least as many decay products as requested
  //exclusive mode: exactly as many decay products as requested
  //(but additional particle types not appearing in the list of requested daughter id's are ignored)
  for (const auto &reqpair : requestedDaughters_) {
    int reqid = reqpair.first;
    int reqcount = reqpair.second;

    int obscount = 0;
    for (const auto &obspair : observedDaughters_) {
      int obsid = obspair.first;

      if (obsid == reqid) {
        obscount = obspair.second;
        break;
      }
    }

    //inclusive criteria not satisfied, veto event
    if (obscount < reqcount)
      return true;

    //exclusive criteria not satisfied, veto event
    if (exclusive_ && obscount > reqcount)
      return true;
  }

  if (matching_ && !remainingDecays_.empty()) {
    return true;
  }

  //all criteria satisfied, don't veto
  return false;
}

void ResonanceDecayFilterHook::resetEventCounter() { counter_event_ = 0; }
