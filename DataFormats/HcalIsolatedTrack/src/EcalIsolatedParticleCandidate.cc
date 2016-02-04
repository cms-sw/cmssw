#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidate.h"

//using namespace reco;

reco::EcalIsolatedParticleCandidate::~EcalIsolatedParticleCandidate() { }

reco::EcalIsolatedParticleCandidate * reco::EcalIsolatedParticleCandidate::clone() const { 
  return new reco::EcalIsolatedParticleCandidate( * this ); 
}

l1extra::L1JetParticleRef reco::EcalIsolatedParticleCandidate::l1TauJet() const {
  return l1tau_;
}


