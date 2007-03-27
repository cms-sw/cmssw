#ifndef RecoTauTag_Pi0Tau_Pi0Fwd_h
#define RecoTauTag_Pi0Tau_Pi0Fwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class Pi0;

  /// collection of Pi0 objects
  typedef std::vector<Pi0> Pi0Collection;

  /// persistent reference to Pi0 objects
  typedef edm::Ref<Pi0Collection> Pi0Ref;

  /// reference to Pi0 collection
  typedef edm::RefProd<Pi0Collection> Pi0RefProd;

  /// vector of references to Pi0 objects all in the same collection
  typedef edm::RefVector<Pi0Collection> Pi0RefVector;

  /// iterator over a vector of references to Pi0 objects
  typedef Pi0RefVector::iterator pi0_iterator;
}

#endif
