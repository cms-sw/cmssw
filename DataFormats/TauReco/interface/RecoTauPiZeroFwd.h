#ifndef DataFormats_TauReco_RecoTauPiZeroFwd_h
#define DataFormats_TauReco_RecoTauPiZeroFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class RecoTauPiZero;
  /// collection of RecoTauPiZero objects
  typedef std::vector<RecoTauPiZero> RecoTauPiZeroCollection;
  /// presistent reference to a RecoTauPiZero
  typedef edm::Ref<RecoTauPiZeroCollection> RecoTauPiZeroRef;
  /// references to RecoTauPiZero collection
  typedef edm::RefProd<RecoTauPiZeroCollection> RecoTauPiZeroRefProd;
  /// vector of references to RecoTauPiZero objects all in the same collection
  typedef edm::RefVector<RecoTauPiZeroCollection> RecoTauPiZeroRefVector;
  /// iterator over a vector of references to RecoTauPiZero objects all in the same collection
  typedef RecoTauPiZeroRefVector::iterator RecoTauPiZeroRefVector_iterator;
}  // namespace reco

#endif
