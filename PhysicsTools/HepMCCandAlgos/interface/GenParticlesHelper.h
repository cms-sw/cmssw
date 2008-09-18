#ifndef PhysicsTools_HepMCCandAlgos_GenParticlesHelper_h
#define PhysicsTools_HepMCCandAlgos_GenParticlesHelper_h

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include <iostream>

namespace GenParticlesHelper {
  
  typedef reco::GenParticleCollection::const_iterator IG;
  typedef reco::GenParticleRefVector::const_iterator IGR;


  /// find all particles of a given pdgId and status
  void findParticles(const reco::GenParticleCollection& sourceParticles,
		     reco::GenParticleRefVector& particleRefs, 
		     int pdgId, int status );

  /// find all descendents of a given status and pdgId
  void findDescendents(const reco::GenParticleRef& base, 
		       reco::GenParticleRefVector& descendents, 
		       int status, int pdgId=0 );

  /// find the particles having the same daughter as baseSister
  void findSisters(const reco::GenParticleRef& baseSister, 
		   reco::GenParticleRefVector& sisterRefs);


  std::ostream& operator<<( std::ostream& out, 
			    const reco::GenParticleRef& genRef );

}

#endif
