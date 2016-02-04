#ifndef DataFormats_TauReco_PFTauFwd_h
#define DataFormats_TauReco_PFTauFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PFTau;
  /// collection of PFTau objects
  typedef std::vector<PFTau> PFTauCollection;
  /// presistent reference to a PFTau
  typedef edm::Ref<PFTauCollection> PFTauRef;
  /// references to PFTau collection
  typedef edm::RefProd<PFTauCollection> PFTauRefProd;
  /// vector of references to PFTau objects all in the same collection
  typedef edm::RefVector<PFTauCollection> PFTauRefVector;
  /// iterator over a vector of references to PFTau objects all in the same collection
  typedef PFTauRefVector::iterator pftau_iterator;
}

#endif
