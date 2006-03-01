#ifndef BTauReco_TrackTagInfoFwd_h
#define BTauReco_TrackTagInfoFwd_h
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"
#include "DataFormats/TrackReco/interface/Track.h"


namespace reco {
  class TrackTagInfo;
  typedef std::vector<TrackTagInfo> TrackTagInfoCollection;
  typedef edm::Ref<TrackTagInfoCollection> TrackTagInfoRef;
  typedef edm::RefVector<TrackTagInfoCollection> TrackTagInfoRefs;
  typedef TrackTagInfoRefs::iterator trackTagInfo_iterator;
}

#endif
