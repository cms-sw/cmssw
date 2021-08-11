#ifndef RecoTracker_MkFit_MkFitSeedWrapper_h
#define RecoTracker_MkFit_MkFitSeedWrapper_h

#include <memory>
#include <vector>

namespace mkfit {
  class Track;
  using TrackVec = std::vector<Track>;
}  // namespace mkfit

class MkFitSeedWrapper {
public:
  MkFitSeedWrapper();
  MkFitSeedWrapper(mkfit::TrackVec seeds);
  ~MkFitSeedWrapper();

  MkFitSeedWrapper(MkFitSeedWrapper const&) = delete;
  MkFitSeedWrapper& operator=(MkFitSeedWrapper const&) = delete;
  MkFitSeedWrapper(MkFitSeedWrapper&&);
  MkFitSeedWrapper& operator=(MkFitSeedWrapper&&);

  mkfit::TrackVec const& seeds() const { return *seeds_; }

private:
  std::unique_ptr<mkfit::TrackVec> seeds_;  // for pimpl pattern
};

#endif
