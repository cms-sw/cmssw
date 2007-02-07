#ifndef BTauReco_JetTagProxyFwd_h
#define BTauReco_JetTagProxyFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class JetTagProxy;
  typedef std::vector<JetTagProxy> JetTagProxyCollection;
  typedef edm::Ref<JetTagProxyCollection> JetTagProxyRef;
  typedef edm::RefProd<JetTagProxyCollection> JetTagProxyRefProd;
  typedef edm::RefVector<JetTagProxyCollection> JetTagProxyRefVector;
}

#endif
