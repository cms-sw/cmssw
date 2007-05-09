#ifndef BTauReco_JTATagInfoFwd_h
#define BTauReco_JTATagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class JTATagInfo;
  typedef std::vector<JTATagInfo> JTATagInfoCollection;
  typedef edm::Ref<JTATagInfoCollection> JTATagInfoRef;
  typedef edm::RefProd<JTATagInfoCollection> JTATagInfoRefProd;
  typedef edm::RefVector<JTATagInfoCollection> JTATagInfoRefVector;
}

#endif
