#ifndef BTauReco_TrackIPTagInfoFwd_h
#define BTauReco_TrackIPTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
namespace reco {
  class TrackIPTagInfo;
  typedef std::vector<TrackIPTagInfo> TrackIPTagInfoCollection;
  typedef edm::Ref<TrackIPTagInfoCollection> TrackIPTagInfoRef;
  typedef edm::RefProd<TrackIPTagInfoCollection> TrackIPTagInfoRefProd;
  typedef edm::RefVector<TrackIPTagInfoCollection> TrackIPTagInfoRefVector;
}

#endif // BTauReco_TrackIPTagInfoFwd_h
