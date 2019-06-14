#ifndef TauReco_PFTauTagInfoFwd_h
#define TauReco_PFTauTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class PFTauTagInfo;
  /// collection of PFTauTagInfo objects
  typedef std::vector<PFTauTagInfo> PFTauTagInfoCollection;
  /// presistent reference to a PFTauTagInfo
  typedef edm::Ref<PFTauTagInfoCollection> PFTauTagInfoRef;
  /// references to PFTauTagInfo collection
  typedef edm::RefProd<PFTauTagInfoCollection> PFTauTagInfoRefProd;
  /// vector of references to PFTauTagInfo objects all in the same collection
  typedef edm::RefVector<PFTauTagInfoCollection> PFTauTagInfoRefVector;
  /// iterator over a vector of references to PFTauTagInfo objects all in the same collection
  typedef PFTauTagInfoRefVector::iterator pftautaginfo_iterator;
}  // namespace reco

#endif
