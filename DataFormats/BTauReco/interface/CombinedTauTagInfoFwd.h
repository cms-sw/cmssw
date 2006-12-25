#ifndef BTauReco_TauCombinationTagInfoFwd_h
#define BTauReco_TauCombinationTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class CombinedTauTagInfo;
  typedef std::vector<CombinedTauTagInfo> CombinedTauTagInfoCollection;
  typedef edm::Ref<CombinedTauTagInfoCollection> CombinedTauTagInfoRef;
  typedef edm::RefProd<CombinedTauTagInfoCollection> CombinedTauTagInfoRefProd;
  typedef edm::RefVector<CombinedTauTagInfoCollection> CombinedTauTagInfoRefVector;
}

#endif
