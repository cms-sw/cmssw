#ifndef BTauReco_PFTauIsolationTagInfoFwd_h
#define BTauReco_PFTauIsolationTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFIsolatedTauTagInfo;
  typedef std::vector<PFIsolatedTauTagInfo> PFIsolatedTauTagInfoCollection;
  typedef edm::Ref<PFIsolatedTauTagInfoCollection> PFIsolatedTauTagInfoRef;
  typedef edm::RefProd<PFIsolatedTauTagInfoCollection> PFIsolatedTauTagInfoRefProd;
  typedef edm::RefVector<PFIsolatedTauTagInfoCollection> PFIsolatedTauTagInfoRefVector;
}

#endif
