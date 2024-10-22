#ifndef HcalIsolatedTrack_EcalIsolatedParticleCandidateFwd_h
#define HcalIsolatedTrack_EcalIsolatedParticleCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class EcalIsolatedParticleCandidate;

  /// collection of EcalIsolatedParticleCandidate objects
  typedef std::vector<EcalIsolatedParticleCandidate> EcalIsolatedParticleCandidateCollection;

  /// reference to an object in a collection of EcalIsolatedParticleCandidate objects
  typedef edm::Ref<EcalIsolatedParticleCandidateCollection> EcalIsolatedParticleCandidateRef;

  /// reference to a collection of EcalIsolatedParticleCandidate objects
  typedef edm::RefProd<EcalIsolatedParticleCandidateCollection> EcalIsolatedParticleCandidateRefProd;

  /// vector of objects in the same collection of EcalIsolatedParticleCandidate objects
  typedef edm::RefVector<EcalIsolatedParticleCandidateCollection> EcalIsolatedParticleCandidateRefVector;

  /// iterator over a vector of reference to EcalIsolatedParticleCandidate objects
  typedef EcalIsolatedParticleCandidateRefVector::iterator EcalIsolatedParticleCandidateIterator;
}  // namespace reco

#endif
