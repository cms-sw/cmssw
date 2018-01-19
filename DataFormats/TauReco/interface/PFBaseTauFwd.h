#ifndef DataFormats_TauReco_PFBaseTauFwd_h
#define DataFormats_TauReco_PFBaseTauFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PFBaseTau;
  /// collection of PFBaseTau objects
  typedef std::vector<PFBaseTau> PFBaseTauCollection;
  /// presistent reference to a PFBaseTau
  typedef edm::Ref<PFBaseTauCollection> PFBaseTauRef;
  /// references to PFBaseTau collection
  typedef edm::RefProd<PFBaseTauCollection> PFBaseTauRefProd;
  /// vector of references to PFBaseTau objects all in the same collection
  typedef edm::RefVector<PFBaseTauCollection> PFBaseTauRefVector;
  /// iterator over a vector of references to PFBaseTau objects all in the same collection
  typedef PFBaseTauRefVector::iterator PFBaseTau_iterator;
}

#endif
