#ifndef TrackReco_RecHitFwd_h
#define TrackReco_RecHitFwd_h
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"

namespace reco {
  class RecHit;
  typedef std::vector<RecHit> RecHitCollection;
  typedef edm::Ref<RecHitCollection> RecHitRef;
  typedef edm::RefVector<RecHitCollection> RecHitRefs;
  typedef RecHitRefs::iterator recHit_iterator;
}

#endif
