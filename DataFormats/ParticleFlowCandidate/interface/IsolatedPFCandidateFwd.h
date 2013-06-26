#ifndef ParticleFlowCandidate_IsolatedPFCandidateFwd_h
#define ParticleFlowCandidate_IsolatedPFCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class IsolatedPFCandidate;

  /// collection of IsolatedPFCandidates
  typedef std::vector<reco::IsolatedPFCandidate> IsolatedPFCandidateCollection;

  /// iterator 
  typedef IsolatedPFCandidateCollection::const_iterator IsolatedPFCandidateConstIterator;

  /// iterator 
  typedef IsolatedPFCandidateCollection::iterator IsolatedPFCandidateIterator;

  /// persistent reference to a IsolatedPFCandidate
  typedef edm::Ref<IsolatedPFCandidateCollection> IsolatedPFCandidateRef;

  /// persistent reference to a IsolatedPFCandidate
  typedef edm::Ptr<IsolatedPFCandidate> IsolatedPFCandidatePtr;  

  /// persistent reference to a IsolatedPFCandidates collection
  typedef edm::RefProd<IsolatedPFCandidateCollection> IsolatedPFCandidateRefProd;

  /// vector of reference to GenParticleCandidate in the same collection
  typedef edm::RefVector<IsolatedPFCandidateCollection> IsolatedPFCandidateRefVector;
  
}

#endif
