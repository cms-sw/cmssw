#ifndef RecoTracker_MkFitCore_standalone_attic_fittestMPlex_h
#define RecoTracker_MkFitCore_standalone_attic_fittestMPlex_h

#include "Event.h"
#include "Track.h"

namespace mkfit {

  double runFittingTestPlex(Event& ev, std::vector<Track>& rectracks);

}  // end namespace mkfit
#endif
