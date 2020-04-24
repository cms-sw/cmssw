#ifndef TrackReco_TrackFwd_h
#define TrackReco_TrackFwd_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

namespace reco
{

class Track;

/// collection of Tracks
typedef std::vector<Track> TrackCollection;

/// persistent reference to a Track
typedef edm::Ref<TrackCollection> TrackRef;

/// persistent reference to a Track collection
typedef edm::RefProd<TrackCollection> TrackRefProd;

/// vector of reference to Track in the same collection
typedef edm::RefVector<TrackCollection> TrackRefVector;

/// iterator over a vector of reference to Track in the same collection
typedef TrackRefVector::iterator track_iterator;

/// persistent reference to a Track, using views
typedef edm::RefToBase<reco::Track> TrackBaseRef;

/// vector of persistent references to a Track, using views
typedef edm::RefToBaseVector<reco::Track> TrackBaseRefVector;

} // namespace reco

#endif

