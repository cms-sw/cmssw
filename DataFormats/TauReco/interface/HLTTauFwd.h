#ifndef DataFormats_TauReco_HLTTauFwd_h
#define DataFormats_TauReco_HLTTauFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class HLTTau;
  /// collection of HLTTau objects
  typedef std::vector<HLTTau> HLTTauCollection;
  /// presistent reference to a HLTTau
  typedef edm::Ref<HLTTauCollection> HLTTauRef;
  /// references to HLTTau collection
  typedef edm::RefProd<HLTTauCollection> HLTTauRefProd;
  /// vector of references to HLTTau objects all in the same collection
  typedef edm::RefVector<HLTTauCollection> HLTTauRefVector;
  /// iterator over a vector of references to HLTTau objects all in the same collection
  typedef HLTTauRefVector::iterator hlttau_iterator;
}  // namespace reco

#endif
