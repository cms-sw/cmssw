#ifndef TauReco_MuonFwd_h
#define TauReco_MuonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Tau;
  /// collection of Tau objects
  typedef std::vector<Tau> TauCollection;
  /// presistent reference to a Tau
  typedef edm::Ref<TauCollection> TauRef;
  /// references to Tau collection
  typedef edm::RefProd<TauCollection> TauRefProd;
  /// vector of references to Tau objects all in the same collection
  typedef edm::RefVector<TauCollection> TauRefVector;
  /// iterator over a vector of references to Tau objects all in the same collection
  typedef TauRefVector::iterator tau_iterator;
}

#endif
