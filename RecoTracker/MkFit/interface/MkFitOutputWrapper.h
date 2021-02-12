#ifndef RecoTracker_MkFit_MkFitOutputWrapper_h
#define RecoTracker_MkFit_MkFitOutputWrapper_h

#include <vector>

namespace mkfit {
  class Track;
  using TrackVec = std::vector<Track>;
}  // namespace mkfit

class MkFitOutputWrapper {
public:
  MkFitOutputWrapper();
  MkFitOutputWrapper(mkfit::TrackVec tracks, bool propagatedToFirstLayer);
  ~MkFitOutputWrapper();

  MkFitOutputWrapper(MkFitOutputWrapper const&) = delete;
  MkFitOutputWrapper& operator=(MkFitOutputWrapper const&) = delete;
  MkFitOutputWrapper(MkFitOutputWrapper&&);
  MkFitOutputWrapper& operator=(MkFitOutputWrapper&&);

  mkfit::TrackVec const& tracks() const { return tracks_; }
  bool propagatedToFirstLayer() const { return propagatedToFirstLayer_; }

private:
  mkfit::TrackVec tracks_;
  bool propagatedToFirstLayer_;
};

#endif
