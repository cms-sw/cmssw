#include "Pythia8/Pythia.h"
#include "GeneratorInterface/Pythia8Interface/interface/ResonanceDecayFilterHook.h"

using namespace Pythia8;


//--------------------------------------------------------------------------
bool ResonanceDecayFilterHook::initAfterBeams() {
  filter_ = settingsPtr->flag("ResonanceDecayFilter:filter");
  exclusive_ = settingsPtr->flag("ResonanceDecayFilter:exclusive");
  eMuAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:eMuAsEquivalent");
  eMuTauAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:eMuTauAsEquivalent");
  allNuAsEquivalent_ = settingsPtr->flag("ResonanceDecayFilter:allNuAsEquivalent");
  mothers_ = settingsPtr->mvec("ResonanceDecayFilter:mothers");
  daughters_ = settingsPtr->mvec("ResonanceDecayFilter:daughters");
  
  
  requestedDaughters_.clear();
  
  for (int id : daughters_) {
    int did = std::abs(id);
    if ( did == 13 && (eMuAsEquivalent_ || eMuTauAsEquivalent_)) {
      did = 11;
    }
    if ( did == 15 && eMuTauAsEquivalent_) {
      did = 11;
    }
    if ( (did == 14 || did == 16) && allNuAsEquivalent_) {
      did = 12;
    }
    
    ++requestedDaughters_[std::abs(did)];
  }
  
  return true;
  
}

//--------------------------------------------------------------------------
bool ResonanceDecayFilterHook::checkVetoResonanceDecays(const Event& process) {
  
  if (!filter_) return false;
    
  observedDaughters_.clear();
  
  //count decay products
  for (int i=0; i<process.size(); ++i) {
    const Particle &p = process[i];
    
    int did = std::abs(p.id());
    
    if ( did == 13 && (eMuAsEquivalent_ || eMuTauAsEquivalent_)) {
      did = 11;
    }
    if ( did == 15 && eMuTauAsEquivalent_) {
      did = 11;
    }
    if ( (did == 14 || did == 16) && allNuAsEquivalent_) {
      did = 12;
    }    
    
    int mid = p.mother1()>0 ? std::abs(process[p.mother1()].id()) : 0;
    
    //if no list of mothers is provided, then all particles
    //in hard process and resonance decays are counted together
    bool validMother = mothers_.size() ? false : true;
    for (int id : mothers_) {
      if (mid == std::abs(id)) {
        validMother = true;
        break;
      }
    }
    
    if (validMother) {
      ++observedDaughters_[did];
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
    if (obscount < reqcount) return true;

    //exclusive criteria not satisfied, veto event
    if (exclusive_ && obscount > reqcount) return true;
    
  }
  
  //all criteria satisfied, don't veto
  return false;
}
