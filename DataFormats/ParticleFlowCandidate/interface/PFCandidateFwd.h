#ifndef ParticleFlowCandidate_PFCandidateFwd_h
#define ParticleFlowCandidate_PFCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class reco::PFCandidate;

  /// collection of PFCandidates
  typedef std::vector<reco::PFCandidate> PFCandidateCollection;

  /// persistent reference to a PFCandidate
  typedef edm::Ref<PFCandidateCollection> PFCandidateRef;

  /// persistent reference to a PFCandidates collection
  typedef edm::RefProd<PFCandidateCollection> PFCandidateRefProd;

  /// vector of reference to GenParticleCandidate in the same collection
  typedef edm::RefVector<PFCandidateCollection> PFCandidateRefVector;

  /// iterator over a vector of reference to GenParticleCandidate in the same collection
  typedef PFCandidateRefVector::iterator PFCandidate_iterator;
}

#endif
