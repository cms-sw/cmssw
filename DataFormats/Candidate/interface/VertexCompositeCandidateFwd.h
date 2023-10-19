#ifndef DataFormats_Candidate_VertexCompositeCandidateFwd_h
#define DataFormats_Candidate_VertexCompositeCandidateFwd_h
#include "DataFormats/Common/interface/OwnVector.h"

namespace reco {
  class VertexCompositeCandidate;
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
  typedef std::vector<VertexCompositeCandidate> VertexCompositeCandidateCollection;
  /// view of a collection containing candidates
  typedef edm::View<VertexCompositeCandidate> VertexCompositeCandidateView;
  /// persistent reference to an object in a collection of Candidate objects
  typedef edm::Ref<VertexCompositeCandidateCollection> VertexCompositeCandidateRef;
  /// vector of references to objects in the same  collection of Candidate objects
  typedef edm::RefVector<VertexCompositeCandidateCollection> VertexCompositeCandidateRefVector;
  /// reference to a collection of Candidate objects
  typedef edm::RefProd<VertexCompositeCandidateCollection> VertexCompositeCandidateRefProd;
  /// vector of references to objects in the same collection of Candidate objects via base type
  typedef edm::RefToBaseProd<VertexCompositeCandidate> VertexCompositeCandidateBaseRefProd;
}  // namespace reco

#endif
