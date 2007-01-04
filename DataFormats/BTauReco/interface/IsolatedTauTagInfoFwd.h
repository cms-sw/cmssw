#ifndef BTauReco_TauIsolationTagInfoFwd_h
#define BTauReco_TauIsolationTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class IsolatedTauTagInfo;
  typedef std::vector<IsolatedTauTagInfo> IsolatedTauTagInfoCollection;
  typedef edm::Ref<IsolatedTauTagInfoCollection> IsolatedTauTagInfoRef;
  typedef edm::RefProd<IsolatedTauTagInfoCollection> IsolatedTauTagInfoRefProd;
  typedef edm::RefVector<IsolatedTauTagInfoCollection> IsolatedTauTagInfoRefVector;
}

#endif
