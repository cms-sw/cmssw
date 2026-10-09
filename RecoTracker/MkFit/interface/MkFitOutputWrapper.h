#ifndef RecoTracker_MkFit_MkFitOutputWrapper_h
#define RecoTracker_MkFit_MkFitOutputWrapper_h

#include <vector>

#include "RecoTracker/MkFitCore/interface/HitStateOnTrack.h"

namespace mkfit {
  class Track;
  using TrackVec = std::vector<Track>;
}  // namespace mkfit

class MkFitOutputWrapper {
public:
  MkFitOutputWrapper();
  MkFitOutputWrapper(mkfit::TrackVec tracks, bool propagatedToFirstLayer);
  // with the final fit's per-hit states (see mkfit::FinalFitHitStates)
  MkFitOutputWrapper(mkfit::TrackVec tracks, bool propagatedToFirstLayer, mkfit::FinalFitHitStates hitStates);
  ~MkFitOutputWrapper();

  MkFitOutputWrapper(MkFitOutputWrapper const&) = delete;
  MkFitOutputWrapper& operator=(MkFitOutputWrapper const&) = delete;
  MkFitOutputWrapper(MkFitOutputWrapper&&);
  MkFitOutputWrapper& operator=(MkFitOutputWrapper&&);

  mkfit::TrackVec const& tracks() const { return tracks_; }
  bool propagatedToFirstLayer() const { return propagatedToFirstLayer_; }
  mkfit::HitStatesOnTracks const& hitStates() const { return hitStates_.smoothed; }
  mkfit::HitStatesOnTracks const& hitStatesFwd() const { return hitStates_.fwd; }
  mkfit::HitStatesOnTracks const& hitStatesBwd() const { return hitStates_.bwd; }
  // the final fit's state at each track's outermost hit (empty: not from the final fit)
  std::vector<mkfit::OuterHitState> const& outerStates() const { return hitStates_.outer; }

private:
  mkfit::TrackVec tracks_;
  bool propagatedToFirstLayer_;
  mkfit::FinalFitHitStates hitStates_;
};

#endif
