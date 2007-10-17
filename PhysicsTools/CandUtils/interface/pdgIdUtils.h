#ifndef PhysicsTools_CandUtils_pdgIdUtils_h
#define PhysicsTools_CandUtils_pdgIdUtils_h
#include "DataFormats/Candidate/interface/Particle.h"

namespace reco {

  inline bool isElectron(const Particle & part) { 
    return abs(part.pdgId())==11;
  }

  inline bool isMuon(const Particle & part) { 
    return abs(part.pdgId())==13;
  }

  inline bool isTau(const Particle & part) { 
    return abs(part.pdgId())==15;
  }

  inline bool isLepton(const Particle & part) { 
    return abs(part.pdgId())==11 || 
      abs(part.pdgId())==13 || 
      abs(part.pdgId())==15; 
  }

  inline bool isNeutrino(const Particle & part) { 
    return abs(part.pdgId())==12 || 
      abs(part.pdgId())==14 || 
      abs(part.pdgId())==16; 
  }

  inline int flavour(const Particle & part) {
    int id = part.pdgId();
    return id/abs(id);
  }

}

#endif
