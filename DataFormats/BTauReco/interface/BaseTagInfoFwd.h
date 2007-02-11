#ifndef BTauReco_BaseTagInfoFwd_h
#define BTauReco_BaseTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class BaseTagInfo;
  typedef std::vector<BaseTagInfo> BaseTagInfoCollection;
  typedef edm::Ref<BaseTagInfoCollection> BaseTagInfoRef;
  typedef edm::RefProd<BaseTagInfoCollection> BaseTagInfoRefProd;
  typedef edm::RefVector<BaseTagInfoCollection> BaseTagInfoRefVector;
}

#endif
