#ifndef TrackInfo_TrackInfoFwd_h
#define TrackInfo_TrackInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class TrackInfo;
  /// collection of TrackInfos
  typedef std::vector<TrackInfo> TrackInfoCollection;

  typedef edm::Ref<TrackInfoCollection> TrackInfoRef;
  
  typedef edm::RefProd<TrackInfoCollection> TrackInfoRefProd;
  
  typedef edm::RefVector<TrackInfoCollection> TrackInfoRefVector;

}

#endif
