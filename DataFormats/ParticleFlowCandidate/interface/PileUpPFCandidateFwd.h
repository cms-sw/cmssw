#ifndef ParticleFlowCandidate_PileUpPFCandidateFwd_h
#define ParticleFlowCandidate_PileUpPFCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PileUpPFCandidate;

  /// collection of PileUpPFCandidates
  typedef std::vector<reco::PileUpPFCandidate> PileUpPFCandidateCollection;

  /// iterator
  typedef PileUpPFCandidateCollection::const_iterator PileUpPFCandidateConstIterator;

  /// iterator
  typedef PileUpPFCandidateCollection::iterator PileUpPFCandidateIterator;

  /// persistent reference to a PileUpPFCandidate
  typedef edm::Ref<PileUpPFCandidateCollection> PileUpPFCandidateRef;

  /// persistent reference to a PileUpPFCandidate
  typedef edm::Ptr<PileUpPFCandidate> PileUpPFCandidatePtr;

  /// persistent reference to a PileUpPFCandidates collection
  typedef edm::RefProd<PileUpPFCandidateCollection> PileUpPFCandidateRefProd;

  /// vector of reference to GenParticleCandidate in the same collection
  typedef edm::RefVector<PileUpPFCandidateCollection> PileUpPFCandidateRefVector;

}  // namespace reco

#endif
