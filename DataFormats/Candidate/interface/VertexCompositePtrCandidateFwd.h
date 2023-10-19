#ifndef DataFormats_Candidate_VertexCompositePtrCandidateFwd_h
#define DataFormats_Candidate_VertexCompositePtrCandidateFwd_h
#include "DataFormats/Common/interface/OwnVector.h"

namespace reco {
  class VertexCompositePtrCandidate;
}

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/View.h"

namespace reco {
  /// collection of Candidate objects
  typedef std::vector<VertexCompositePtrCandidate> VertexCompositePtrCandidateCollection;
  /// view of a collection containing candidates
  typedef edm::View<VertexCompositePtrCandidate> VertexCompositePtrCandidateView;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::Ref<VertexCompositePtrCandidateCollection> VertexCompositePtrCandidateRef;
  //  /// persistent reference to an object in a collection of Candidate objects
  //  typedef edm::RefToBase<VertexCompositePtrCandidate> VertexCompositePtrCandidateBaseRef;
  /// vector of references to objects in the same  collection of Candidate objects
  typedef edm::RefVector<VertexCompositePtrCandidateCollection> VertexCompositePtrCandidateRefVector;
  /// vector of references to objects in the same collection of Candidate objects via base type
  typedef edm::RefToBaseVector<VertexCompositePtrCandidate> VertexCompositePtrCandidateBaseRefVector;
  /// reference to a collection of Candidate objects
  typedef edm::RefProd<VertexCompositePtrCandidateCollection> VertexCompositePtrCandidateRefProd;
  /// vector of references to objects in the same collection of Candidate objects via base type
  typedef edm::RefToBaseProd<VertexCompositePtrCandidate> VertexCompositePtrCandidateBaseRefProd;
}  // namespace reco

#endif
