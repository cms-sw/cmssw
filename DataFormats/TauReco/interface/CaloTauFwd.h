#ifndef DataFormats_TauReco_CaloTauFwd_h
#define DataFormats_TauReco_CaloTauFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class CaloTau;
  /// collection of CaloTau objects
  typedef std::vector<CaloTau> CaloTauCollection;
  /// presistent reference to a CaloTau
  typedef edm::Ref<CaloTauCollection> CaloTauRef;
  /// references to CaloTau collection
  typedef edm::RefProd<CaloTauCollection> CaloTauRefProd;
  /// vector of references to CaloTau objects all in the same collection
  typedef edm::RefVector<CaloTauCollection> CaloTauRefVector;
  /// iterator over a vector of references to CaloTau objects all in the same collection
  typedef CaloTauRefVector::iterator calotau_iterator;
}

#endif
