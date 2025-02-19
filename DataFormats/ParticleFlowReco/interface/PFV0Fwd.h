#ifndef ParticleFlowReco_PFV0Fwd_h
#define ParticleFlowReco_PFV0Fwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFV0;

  /// collection of PFV0 objects
  typedef std::vector<PFV0> PFV0Collection;

  /// persistent reference to PFV0 objects
  typedef edm::Ref<PFV0Collection> PFV0Ref;

  /// reference to PFV0 collection
  typedef edm::RefProd<PFV0Collection> PFV0RefProd;

  /// vector of references to PFV0 objects all in the same collection
  typedef edm::RefVector<PFV0Collection> PFV0RefVector;

  /// iterator over a vector of references to PFV0 objects
  typedef PFV0RefVector::iterator PFV0_iterator;
}

#endif
