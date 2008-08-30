#ifndef ParticleFlowCandidate_PFCandidateFwd_h
#define ParticleFlowCandidate_PFCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PFCandidate;

  /// collection of PFCandidates
  typedef std::vector<reco::PFCandidate> PFCandidateCollection;

  /// iterator 
  typedef PFCandidateCollection::const_iterator PFCandidateConstIterator;

  /// iterator 
  typedef PFCandidateCollection::iterator PFCandidateIterator;

  /// persistent reference to a PFCandidate
  typedef edm::Ref<PFCandidateCollection> PFCandidateRef;

  /// persistent reference to a PFCandidates collection
  typedef edm::RefProd<PFCandidateCollection> PFCandidateRefProd;

  /// vector of reference to GenParticleCandidate in the same collection
  typedef edm::RefVector<PFCandidateCollection> PFCandidateRefVector;
  
}

#endif
