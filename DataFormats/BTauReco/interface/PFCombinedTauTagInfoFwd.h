#ifndef BTauReco_PFTauCombinationTagInfoFwd_h
#define BTauReco_PFTauCombinationTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFCombinedTauTagInfo;
  typedef std::vector<PFCombinedTauTagInfo> PFCombinedTauTagInfoCollection;
  typedef edm::Ref<PFCombinedTauTagInfoCollection> PFCombinedTauTagInfoRef;
  typedef edm::RefProd<PFCombinedTauTagInfoCollection> PFCombinedTauTagInfoRefProd;
  typedef edm::RefVector<PFCombinedTauTagInfoCollection> PFCombinedTauTagInfoRefVector;
}

#endif
