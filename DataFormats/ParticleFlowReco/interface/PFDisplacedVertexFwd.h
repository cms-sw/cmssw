#ifndef ParticleFlowReco_PFDisplacedVertexFwd_h
#define ParticleFlowReco_PFDisplacedVertexFwd_h
#include <vector>

#include "DataFormats/Common/interface/Ref.h"
/* #include "DataFormats/Common/interface/RefVector.h" */
/* #include "DataFormats/Common/interface/RefProd.h" */

namespace reco {
  class PFDisplacedVertex;

  /// collection of PFDisplacedVertex objects
  typedef std::vector<PFDisplacedVertex> PFDisplacedVertexCollection;

  /// persistent reference to a PFDisplacedVertex objects
  typedef edm::Ref<PFDisplacedVertexCollection> PFDisplacedVertexRef;

  /// handle to a PFDisplacedVertex collection
  typedef edm::Handle<PFDisplacedVertexCollection> PFDisplacedVertexHandle;

  /// iterator over a vector of references to PFDisplacedVertex objects
  /*   typedef PFDisplacedVertexRefVector::iterator PFDisplacedVertex_iterator; */
}  // namespace reco

#endif
