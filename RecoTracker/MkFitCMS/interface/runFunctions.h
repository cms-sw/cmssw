#ifndef RecoTracker_MkFitCMS_interface_runFunctions_h
#define RecoTracker_MkFitCMS_interface_runFunctions_h

#include "RecoTracker/MkFitCore/interface/Track.h"

#include "RecoTracker/MkFitCore/interface/HitStructures.h"

namespace mkfit {

  class IterationConfig;
  class MkBuilder;

  // nullptr is a valid mask ... means no mask for these layers.
  void run_OneIteration(const TrackerInfo &trackerInfo,
                        const IterationConfig &itconf,
                        const EventOfHits &eoh,
                        const std::vector<const std::vector<bool> *> &hit_masks,
                        MkBuilder &builder,
                        TrackVec &seeds,
                        TrackVec &out_tracks,
                        bool do_seed_clean,
                        bool do_backward_fit,
                        bool do_remove_duplicates);

}  // end namespace mkfit
#endif
