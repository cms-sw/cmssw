#ifndef BTauReco_TrackWithTagInfoFwd_h
#define BTauReco_TrackWithTagInfoFwd_h
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"
#include "DataFormats/TrackReco/interface/Track.h"


namespace reco {
  class TrackWithTagInfo;
  typedef std::vector<TrackWithTagInfo> TrackWithTagInfoCollection;
  typedef edm::Ref<TrackWithTagInfoCollection> TrackWithTagInfoRef;
  typedef edm::RefVector<TrackWithTagInfoCollection> TrackWithTagInfoRefs;
  typedef TrackWithTagInfoRefs::iterator trackWithTagInfo_iterator;
}

#endif
