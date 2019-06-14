#ifndef ParticleFlowReco_PFBlockFwd_h
#define ParticleFlowReco_PFBlockFwd_h
#include <vector>

#include "DataFormats/Common/interface/Ref.h"
/* #include "DataFormats/Common/interface/RefVector.h" */
/* #include "DataFormats/Common/interface/RefProd.h" */

namespace reco {
  class PFBlock;

  /// collection of PFBlock objects
  typedef std::vector<PFBlock> PFBlockCollection;

  /// collection of PFBlock objects
  typedef std::vector<PFBlock> PFBlockCollection;

  /// persistent reference to PFCluster objects
  typedef edm::Ref<PFBlockCollection> PFBlockRef;

  /// handle to a block collection
  typedef edm::Handle<PFBlockCollection> PFBlockHandle;

  /// iterator over a vector of references to PFBlock objects
  /*   typedef PFBlockRefVector::iterator PFBlock_iterator; */
}  // namespace reco

#endif
