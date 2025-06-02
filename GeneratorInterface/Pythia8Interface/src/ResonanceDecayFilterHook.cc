#include "Pythia8/Pythia.h"
#include "GeneratorInterface/Pythia8Interface/interface/ResonanceDecayFilterHook.h"

using namespace Pythia8;

//--------------------------------------------------------------------------
int ResonanceDecayFilterHook::idCat(int id) {
  id = abs(id);
  if (id == 13 && (eMuAsEquivalent_ || eMuTauAsEquivalent_))
    id = 11;
  else if (id == 15 && eMuTauAsEquivalent_)
    id = 11;
  else if ((id == 14 || id == 16) && allNuAsEquivalent_)
    id = 12;
  else if ((id == 2 || id == 3 || id == 4) && udscAsEquivalent_)
    id = 1;
  else if ((id == 2 || id == 3 || id == 4 || id == 5) && udscbAsEquivalent_)
    id = 1;
  else if ((id == 23 || id == 24) && wzAsEquivalent_)
    id = 23;
  return id;
}

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
  matchedDecays_ = settingsPtr->wvec("ResonanceDecayFilter:matchedDecays");

  requestedDaughters_.clear();

  for (int id : daughters_) {
    int did = ResonanceDecayFilterHook::idCat(id);
    ++requestedDaughters_[std::abs(did)];
  }

  requestedDecays_.clear();

  if (matching_) {
    if (matchedDecays_.empty()) {
      std::cerr << "You must indicate a non-empty matched decays list for matching mode." << std::endl;
    }

    for (const std::string &decayStr : matchedDecays_) {
      size_t colonPos = decayStr.find(':');
      if (colonPos == std::string::npos) {
        std::cerr << "Malformed decay string (no colon): " << decayStr << std::endl;
        continue;
      }
      std::string motherStr = decayStr.substr(0, colonPos);
      std::string daughtersStr = decayStr.substr(colonPos + 1);

      int motherID = std::abs(std::stoi(motherStr));

      std::istringstream iss(daughtersStr);
      std::string token;
      while (iss >> token) {
        try {
          int daughterID = ResonanceDecayFilterHook::idCat(std::stoi(token));
          requestedDecays_.insert({motherID, daughterID});
        } catch (const std::invalid_argument &e) {
          std::cerr << "Invalid daughter ID in decay string: " << decayStr << std::endl;
        }
      }
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

    int did = ResonanceDecayFilterHook::idCat(p.id());

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
