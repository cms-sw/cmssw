#ifndef DataFormats_TauReco_PFTauDecayModeFwd_h
#define DataFormats_TauReco_PFTauDecayModeFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PFTauDecayMode;
  /// collection of PFTauDecayMode objects
  typedef std::vector<PFTauDecayMode> PFTauDecayModeCollection;
  /// presistent reference to a PFTauDecayMode
  typedef edm::Ref<PFTauDecayModeCollection> PFTauDecayModeRef;
  /// references to PFTauDecayMode collection
  typedef edm::RefProd<PFTauDecayModeCollection> PFTauDecayModeRefProd;
  /// vector of references to PFTauDecayMode objects all in the same collection
  typedef edm::RefVector<PFTauDecayModeCollection> PFTauDecayModeRefVector;
  /// iterator over a vector of references to PFTauDecayMode objects all in the same collection
  typedef PFTauDecayModeRefVector::iterator pftaudecaymode_iterator;
}  // namespace reco

#endif
