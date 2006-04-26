#ifndef BTauReco_JetTagFwd_h
#define BTauReco_JetTagFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class JetTag;
  typedef std::vector<JetTag> JetTagCollection;
  typedef edm::Ref<JetTagCollection> JetTagRef;
  typedef edm::RefProd<JetTagCollection> JetTagRefProd;
  typedef edm::RefVector<JetTagCollection> JetTagRefVector;
}

#endif
