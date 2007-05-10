#ifndef BTauReco_JetTagInfoFwd_h
#define BTauReco_JetTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class JetTagInfo;
  typedef std::vector<JetTagInfo> JetTagInfoCollection;
  typedef edm::Ref<JetTagInfoCollection> JetTagInfoRef;
  typedef edm::RefProd<JetTagInfoCollection> JetTagInfoRefProd;
  typedef edm::RefVector<JetTagInfoCollection> JetTagInfoRefVector;
}

#endif
