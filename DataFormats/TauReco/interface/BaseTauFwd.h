#ifndef DataFormats_TauReco_BaseTauFwd_h
#define DataFormats_TauReco_BaseTauFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class BaseTau;
  /// collection of BaseTau objects
  typedef std::vector<BaseTau> BaseTauCollection;
  /// presistent reference to a BaseTau
  typedef edm::Ref<BaseTauCollection> BaseTauRef;
  /// references to BaseTau collection
  typedef edm::RefProd<BaseTauCollection> BaseTauRefProd;
  /// vector of references to BaseTau objects all in the same collection
  typedef edm::RefVector<BaseTauCollection> BaseTauRefVector;
  /// iterator over a vector of references to BaseTau objects all in the same collection
  typedef BaseTauRefVector::iterator basetau_iterator;
}  // namespace reco

#endif
