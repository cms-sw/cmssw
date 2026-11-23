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

  using clean_duplicates_cf = void(TrackVec &, const IterationConfig &, const TrackerInfo &, const BeamSpot &);
  using clean_duplicates_func = std::function<clean_duplicates_cf>;

  // What a track scorer sees of a track or candidate. Hole counts are always the
  // real ones; penalize_tail_holes says whether the caller wants the tail holes
  // charged. score_in is the score the object carries when the scorer is called
  // (for MkFinderV2p2 at the final pick, the summed layer-step log-likelihood of
  // the search), 0 where it carries none.
  struct TrackScoreInput {
    int n_found_hits = 0;
    int n_tail_holes = 0;
    int n_overlap_hits = 0;
    int n_inside_holes = 0;
    int n_seed_hits = 0;
    float chi2 = 0;
    float pt = 0;
    float score_in = 0;
    bool penalize_tail_holes = false;
    bool in_find_candidates = false;
  };

  using track_score_cf = float(const TrackScoreInput &);
  using track_score_func = std::function<track_score_cf>;

  using cpe_cf = bool(int orig_hit_idx, float ltp_arr[6], float (&hit_arr)[5]);
  using cpe_func = std::function<cpe_cf>;

}  // end namespace mkfit

#endif
