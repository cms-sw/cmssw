#ifndef RecoTracker_MkFitCore_standalone_attic_seedtestMPlex_h
#define RecoTracker_MkFitCore_standalone_attic_seedtestMPlex_h

#include "Event.h"
#include "Track.h"
#include "HitStructures.h"

namespace mkfit {

  void findSeedsByRoadSearch(TripletIdxConVec& seed_idcs,
                             std::vector<LayerOfHits>& evt_lay_hits,
                             int lay1_size,
                             Event*& ev);

}  // end namespace mkfit
#endif
