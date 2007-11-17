#ifndef HepMCCandidate_GenParticleCAndidateFwd_h
#define HepMCCandidate_GenParticleCAndidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class GenParticleCandidate;
  /// collection of GenParticleCandidates
  typedef std::vector<GenParticleCandidate> GenParticleCandidateCollection;
  /// persistent reference to a GenParticleCandidate
  typedef edm::Ref<GenParticleCandidateCollection> GenParticleCandidateRef;
  /// persistent reference to a GenParticleCandidate collection
  typedef edm::RefProd<GenParticleCandidateCollection> GenParticleCandidateRefProd;
  /// vector of reference to GenParticleCandidate in the same collection
  typedef edm::RefVector<GenParticleCandidateCollection> GenParticleCandidateRefVector;
  /// iterator over a vector of reference to GenParticleCandidate in the same collection
  typedef GenParticleCandidateRefVector::iterator GenParticleCandidate_iterator;
}

#endif
