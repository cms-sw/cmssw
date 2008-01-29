#ifndef HepMCCandidate_GenParticleFwd_h
#define HepMCCandidate_GenParticleFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Association.h"

namespace reco {
  class GenParticle;
  /// collection of GenParticles
  typedef std::vector<GenParticle> GenParticleCollection;
  /// persistent reference to a GenParticle
  typedef edm::Ref<GenParticleCollection> GenParticleRef;
  /// persistent reference to a GenParticle collection
  typedef edm::RefProd<GenParticleCollection> GenParticleRefProd;
  /// vector of reference to GenParticle in the same collection
  typedef edm::RefVector<GenParticleCollection> GenParticleRefVector;
  /// vector of reference to GenParticle in the same collection
  typedef edm::Association<GenParticleCollection> GenParticleMatch;
}

#endif
