#ifndef TrackReco_TrackExtraFwd_h
#define TrackReco_TrackExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class TrackExtra;
  typedef std::vector<TrackExtra> TrackExtraCollection;
  typedef edm::Ref<TrackExtraCollection> TrackExtraRef;
  typedef edm::RefVector<TrackExtraCollection> TrackExtraRefs;
  typedef TrackExtraRefs::iterator trackExtra_iterator;
}

#endif
