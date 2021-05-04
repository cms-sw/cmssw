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
  MkFitOutputWrapper(mkfit::TrackVec&& candidateTracks, mkfit::TrackVec&& fitTracks);
  ~MkFitOutputWrapper();

  MkFitOutputWrapper(MkFitOutputWrapper const&) = delete;
  MkFitOutputWrapper& operator=(MkFitOutputWrapper const&) = delete;
  MkFitOutputWrapper(MkFitOutputWrapper&&);
  MkFitOutputWrapper& operator=(MkFitOutputWrapper&&);

  mkfit::TrackVec const& candidateTracks() const { return candidateTracks_; }
  mkfit::TrackVec const& fitTracks() const { return fitTracks_; }

private:
  mkfit::TrackVec candidateTracks_;
  mkfit::TrackVec fitTracks_;
};

#endif
