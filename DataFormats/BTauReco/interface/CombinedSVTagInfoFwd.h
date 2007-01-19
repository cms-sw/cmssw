#ifndef BTauReco_CombinedSVTagInfoFwd_h
#define BTauReco_CombinedSVTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class CombinedSVTagInfo;
  typedef std::vector<CombinedSVTagInfo> CombinedSVTagInfoCollection;
  typedef edm::Ref<CombinedSVTagInfoCollection> CombinedSVTagInfoRef;
  typedef edm::RefProd<CombinedSVTagInfoCollection> CombinedSVTagInfoRefProd;
  typedef edm::RefVector<CombinedSVTagInfoCollection> CombinedSVTagInfoRefVector;
}

#endif // BTauReco_CombinedSVTagInfoFwd_h
