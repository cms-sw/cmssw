#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidate.h"

using namespace reco;

EcalIsolatedParticleCandidate::~EcalIsolatedParticleCandidate() { }

EcalIsolatedParticleCandidate * EcalIsolatedParticleCandidate::clone() const { 
  return new EcalIsolatedParticleCandidate( * this ); 
}

SuperClusterRef EcalIsolatedParticleCandidate::superCluster() const {
  return superClu_;
}


bool EcalIsolatedParticleCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 &&  checkOverlap( superCluster(), o->superCluster() ) );
}

