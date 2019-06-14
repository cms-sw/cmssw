#ifndef ParticleFlowReco_PFDisplacedVertexCandidateFwd_h
#define ParticleFlowReco_PFDisplacedVertexCandidateFwd_h
#include <vector>

#include "DataFormats/Common/interface/Ref.h"
/* #include "DataFormats/Common/interface/RefVector.h" */
/* #include "DataFormats/Common/interface/RefProd.h" */

namespace reco {
  class PFDisplacedVertexCandidate;

  /// collection of PFDisplacedVertexCandidate objects
  typedef std::vector<PFDisplacedVertexCandidate> PFDisplacedVertexCandidateCollection;

  /// persistent reference to a PFDisplacedVertexCandidate objects
  typedef edm::Ref<PFDisplacedVertexCandidateCollection> PFDisplacedVertexCandidateRef;

  /// handle to a PFDisplacedVertexCandidate collection
  typedef edm::Handle<PFDisplacedVertexCandidateCollection> PFDisplacedVertexCandidateHandle;

}  // namespace reco

#endif
