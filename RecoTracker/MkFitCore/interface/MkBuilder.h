#ifndef RecoTracker_MkFitCore_interface_MkBuilder_h
#define RecoTracker_MkFitCore_interface_MkBuilder_h

#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"
#include "RecoTracker/MkFitCore/interface/MkJob.h"

#include <atomic>
#include <functional>
#include <map>
#include <vector>

namespace mkfit {

  class CandCloner;
  class LayerInfo;
  class MkFinder;
  class MkFitter;
  class TrackerInfo;

  class Event;

  //==============================================================================
  // MkBuilder
  //==============================================================================

  class MkBuilder {
  public:
    using insert_seed_foo = void(const Track &, int, int);

    typedef std::vector<std::pair<int, int>> CandIdx_t;

    MkBuilder(bool silent = true) : m_silent(silent) {}
    ~MkBuilder() = default;

    // --------

    static std::unique_ptr<MkBuilder> make_builder(bool silent = true);
    static void populate();

    int total_cands() const;
    std::pair<int, int> max_hits_layer(const EventOfHits &eoh) const;

    // --------

    void begin_event(MkJob *job, Event *ev, const char *build_type);
    void end_event();
    void release_memory();

    void import_seeds(const TrackVec &in_seeds, const bool seeds_sorted, std::function<insert_seed_foo> insert_seed);

    // filter for rearranging cands that will / will not do backward search.
    int filter_comb_cands(filter_candidates_func filter, bool attempt_all_cands);

    void find_min_max_hots_size();

    void select_best_comb_cands(bool clear_m_tracks = false, bool remove_missing_hits = false);
    void export_best_comb_cands(TrackVec &out_vec, bool remove_missing_hits = false);
    void export_tracks(TrackVec &out_vec);

    void compactifyHitStorageForBestCand(bool remove_seed_hits, int backward_fit_min_hits) {
      m_event_of_comb_cands.compactifyHitStorageForBestCand(remove_seed_hits, backward_fit_min_hits);
    }

    void beginBkwSearch() { m_event_of_comb_cands.beginBkwSearch(); }
    void endBkwSearch() { m_event_of_comb_cands.endBkwSearch(); }

    // MIMI hack to export tracks for BH
    const TrackVec &ref_tracks() const { return m_tracks; }
    TrackVec &ref_tracks_nc() { return m_tracks; }

    const EventOfCombCandidates &ref_eocc() const { return m_event_of_comb_cands; }

    // --------

    void find_tracks_load_seeds_BH(const TrackVec &in_seeds, const bool seeds_sorted);  // for FindTracksBestHit
    void find_tracks_load_seeds(const TrackVec &in_seeds, const bool seeds_sorted);

    int find_tracks_unroll_candidates(std::vector<std::pair<int, int>> &seed_cand_vec,
                                      int start_seed,
                                      int end_seed,
                                      int layer,
                                      int prev_layer,
                                      bool pickup_only,
                                      SteeringParams::IterationType_e iteration_dir);

    void find_tracks_handle_missed_layers(MkFinder *mkfndr,
                                          const LayerInfo &layer_info,
                                          std::vector<std::vector<TrackCand>> &tmp_cands,
                                          const std::vector<std::pair<int, int>> &seed_cand_idx,
                                          const int region,
                                          const int start_seed,
                                          const int itrack,
                                          const int end);

    void find_tracks_in_layers(CandCloner &cloner,
                               MkFinder *mkfndr,
                               SteeringParams::IterationType_e iteration_dir,
                               const int start_seed,
                               const int end_seed,
                               const int region);

    // --------

    void seed_post_cleaning(TrackVec &tv);

    void findTracksBestHit(SteeringParams::IterationType_e iteration_dir = SteeringParams::IT_FwdSearch);
    void findTracksStandard(SteeringParams::IterationType_e iteration_dir = SteeringParams::IT_FwdSearch);
    void findTracksCloneEngine(SteeringParams::IterationType_e iteration_dir = SteeringParams::IT_FwdSearch);

    void backwardFitBH();
    void fit_cands_BH(MkFinder *mkfndr, int start_cand, int end_cand, int region);

    void backwardFit();
    void fit_cands(MkFinder *mkfndr, int start_cand, int end_cand, int region);

  private:
    void fit_one_seed_set(TrackVec &simtracks, int itrack, int end, MkFitter *mkfttr, const bool is_brl[]);

    MkJob *m_job = nullptr;

    // MIMI -- Used by seed processing / validation.
    Event *m_event = nullptr;

    // State for BestHit
    TrackVec m_tracks;

    // State for Std / CloneEngine
    EventOfCombCandidates m_event_of_comb_cands;

    // Per-region seed information
    std::vector<int> m_seedEtaSeparators;
    std::vector<int> m_seedMinLastLayer;
    std::vector<int> m_seedMaxLastLayer;

    std::atomic<int> m_nan_n_silly_per_layer_count;

    bool m_silent;
  };

}  // end namespace mkfit

#endif
