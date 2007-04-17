#ifndef BTauReco_EMIsolationTagInfoFwd_h
#define BTauReco_EMIsolationTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class EMIsolatedTauTagInfo;
  typedef std::vector<EMIsolatedTauTagInfo> EMIsolatedTauTagInfoCollection;
  typedef edm::Ref<EMIsolatedTauTagInfoCollection> EMIsolatedTauTagInfoRef;
  typedef edm::RefProd<EMIsolatedTauTagInfoCollection> EMIsolatedTauTagInfoRefProd;
  typedef edm::RefVector<EMIsolatedTauTagInfoCollection> EMIsolatedTauTagInfoRefVector;
}

#endif
