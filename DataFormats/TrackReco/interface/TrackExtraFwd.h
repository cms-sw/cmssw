#ifndef TrackReco_TrackExtraFwd_h
#define TrackReco_TrackExtraFwd_h
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"

namespace reco {
  class TrackExtra;
  typedef std::vector<TrackExtra> TrackExtraCollection;
  typedef edm::Ref<TrackExtraCollection> TrackExtraRef;
  typedef edm::RefVector<TrackExtraCollection> TrackExtraRefs;
  typedef TrackExtraRefs::iterator trackExtra_iterator;
}

#endif
