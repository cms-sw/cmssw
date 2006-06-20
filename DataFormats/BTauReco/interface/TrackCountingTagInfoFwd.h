#ifndef BTauReco_TrackCountingTagInfoFwd_h
#define BTauReco_TrackCountingTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class TrackCountingTagInfo;
  typedef std::vector<TrackCountingTagInfo> TrackCountingTagInfoCollection;
  typedef edm::Ref<TrackCountingTagInfoCollection> TrackCountingTagInfoRef;
  typedef edm::RefProd<TrackCountingTagInfoCollection> TrackCountingTagInfoRefProd;
  typedef edm::RefVector<TrackCountingTagInfoCollection> TrackCountingTagInfoRefVector;
}

#endif // BTauReco_TrackCountingTagInfoFwd_h
