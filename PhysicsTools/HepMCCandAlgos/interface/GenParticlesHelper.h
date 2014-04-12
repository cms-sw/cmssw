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

  /// find all descendents of a given status and pdgId (recursive)
  void findDescendents(const reco::GenParticleRef& base, 
		       reco::GenParticleRefVector& descendents, 
		       int status, int pdgId=0 );

  /// find the particles having the same daughter as baseSister
  void findSisters(const reco::GenParticleRef& baseSister, 
		   reco::GenParticleRefVector& sisterRefs);

  /// does the particle have an ancestor with this pdgId and this status? 
  bool hasAncestor( const reco::GenParticle* particle,
		    int pdgId, int status );

  /// check if particle is direct (has status 3 or is a daughter of particle with status 3)
  bool isDirect(const reco::GenParticleRef& particle);

  std::ostream& operator<<( std::ostream& out, 
			    const reco::GenParticleRef& genRef );

}

#endif
