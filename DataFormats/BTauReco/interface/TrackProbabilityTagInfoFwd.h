#ifndef BTauReco_TrackProbabilityTagInfoFwd_h
#define BTauReco_TrackProbabilityTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class TrackProbabilityTagInfo;
  typedef std::vector<TrackProbabilityTagInfo> TrackProbabilityTagInfoCollection;
  typedef edm::Ref<TrackProbabilityTagInfoCollection> TrackProbabilityTagInfoRef;
  typedef edm::RefProd<TrackProbabilityTagInfoCollection> TrackProbabilityTagInfoRefProd;
  typedef edm::RefVector<TrackProbabilityTagInfoCollection> TrackProbabilityTagInfoRefVector;
}

#endif // BTauReco_TrackProbabilityTagInfoFwd_h
