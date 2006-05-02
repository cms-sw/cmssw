#ifndef BTauReco_CombinedBTagInfoFwd_h
#define BTauReco_CombinedBTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class CombinedBTagInfo;
  typedef std::vector<CombinedBTagInfo> CombinedBTagInfoCollection;
  typedef edm::Ref<CombinedBTagInfoCollection> CombinedBTagInfoCollectionRef;
  typedef edm::RefProd<CombinedBTagInfoCollection> CombinedBTagInfoRefCollectionProd;
  typedef edm::RefVector<CombinedBTagInfoCollection> CombinedBTagInfoCollectionRefVector;
}

#endif

