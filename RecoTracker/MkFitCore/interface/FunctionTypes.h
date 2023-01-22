#ifndef RecoTracker_MkFitCore_interface_FunctionTypes_h
#define RecoTracker_MkFitCore_interface_FunctionTypes_h

#include <functional>

namespace mkfit {

  struct BeamSpot;
  class EventOfHits;
  class TrackerInfo;
  class Track;
  class TrackCand;
  class MkJob;
  class IterationConfig;
  class IterationSeedPartition;

  typedef std::vector<Track> TrackVec;

  // ----------------------------------------------------------

  using clean_seeds_cf = int(TrackVec &, const IterationConfig &, const BeamSpot &);
  using clean_seeds_func = std::function<clean_seeds_cf>;

  using partition_seeds_cf = void(const TrackerInfo &, const TrackVec &, const EventOfHits &, IterationSeedPartition &);
  using partition_seeds_func = std::function<partition_seeds_cf>;

  using filter_candidates_cf = bool(const TrackCand &, const MkJob &);
  using filter_candidates_func = std::function<filter_candidates_cf>;

  using clean_duplicates_cf = void(TrackVec &, const IterationConfig &);
  using clean_duplicates_func = std::function<clean_duplicates_cf>;

  using track_score_cf = float(const int nfoundhits,
                               const int ntailholes,
                               const int noverlaphits,
                               const int nmisshits,
                               const float chi2,
                               const float pt,
                               const bool inFindCandidates);
  using track_score_func = std::function<track_score_cf>;

}  // end namespace mkfit

#endif
