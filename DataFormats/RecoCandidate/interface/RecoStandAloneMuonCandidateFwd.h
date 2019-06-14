#ifndef RecoCandidate_RecoStandAloneMuonCandidateFwd_h
#define RecoCandidate_RecoStandAloneMuonCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class RecoStandAloneMuonCandidate;

  /// collectin of RecoStandAloneMuonCandidate objects
  typedef std::vector<RecoStandAloneMuonCandidate> RecoStandAloneMuonCandidateCollection;

  /// reference to an object in a collection of RecoStandAloneMuonCandidate objects
  typedef edm::Ref<RecoStandAloneMuonCandidateCollection> RecoStandAloneMuonCandidateRef;

  /// reference to a collection of RecoStandAloneMuonCandidate objects
  typedef edm::RefProd<RecoStandAloneMuonCandidateCollection> RecoStandAloneMuonCandidateRefProd;

  /// vector of objects in the same collection of RecoStandAloneMuonCandidate objects
  typedef edm::RefVector<RecoStandAloneMuonCandidateCollection> RecoStandAloneMuonCandidateRefVector;

  /// iterator over a vector of reference to RecoStandAloneMuonCandidate objects
  typedef RecoStandAloneMuonCandidateRefVector::iterator recoStandAloneMuonCandidate_iterator;
}  // namespace reco

#endif
