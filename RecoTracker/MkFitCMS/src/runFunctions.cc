#include "RecoTracker/MkFitCMS/interface/runFunctions.h"
#include "RecoTracker/MkFitCore/interface/MkBuilder.h"
#include "RecoTracker/MkFitCMS/interface/MkStdSeqs.h"

#include "oneapi/tbb/parallel_for.h"

#include <memory>

namespace mkfit {

  //==============================================================================
  // run_OneIteration
  //
  // One-stop function for running track building from CMSSW.
  //==============================================================================

  struct IterationMaskIfcCmssw : public IterationMaskIfcBase {
    const TrackerInfo &m_trk_info;
    const std::vector<const std::vector<bool> *> &m_mask_vector;

    IterationMaskIfcCmssw(const TrackerInfo &ti, const std::vector<const std::vector<bool> *> &maskvec)
        : m_trk_info(ti), m_mask_vector(maskvec) {}

    const std::vector<bool> *get_mask_for_layer(int layer) const override {
      return m_trk_info.layer(layer).is_pixel() ? m_mask_vector[0] : m_mask_vector[1];
    }
  };

  void run_OneIteration(const TrackerInfo &trackerInfo,
                        const IterationConfig &itconf,
                        const EventOfHits &eoh,
                        const std::vector<const std::vector<bool> *> &hit_masks,
                        MkBuilder &builder,
                        TrackVec &seeds,
                        TrackVec &out_tracks,
                        bool do_seed_clean,
                        bool do_backward_fit,
                        bool do_remove_duplicates) {
    IterationMaskIfcCmssw it_mask_ifc(trackerInfo, hit_masks);

    MkJob job({trackerInfo, itconf, eoh, eoh.refBeamSpot(), &it_mask_ifc});

    builder.begin_event(&job, nullptr, __func__);

    // Seed cleaning not done on all iterations.
    do_seed_clean = do_seed_clean && itconf.m_seed_cleaner;

    if (do_seed_clean)
      itconf.m_seed_cleaner(seeds, itconf, eoh.refBeamSpot());

    // Check nans in seeds -- this should not be needed when Slava fixes
    // the track parameter coordinate transformation.
    builder.seed_post_cleaning(seeds);

    if (itconf.m_requires_seed_hit_sorting) {
      for (auto &s : seeds)
        s.sortHitsByLayer();  // sort seed hits for the matched hits (I hope it works here)
    }

    builder.find_tracks_load_seeds(seeds, do_seed_clean);

    builder.findTracksCloneEngine();

    // Pre backward-fit filtering.
    filter_candidates_func pre_filter;
    if (do_backward_fit && itconf.m_pre_bkfit_filter)
      pre_filter = [&](const TrackCand &tc, const MkJob &jb) -> bool {
        return itconf.m_pre_bkfit_filter(tc, jb) && StdSeq::qfilter_nan_n_silly<TrackCand>(tc, jb);
      };
    else if (itconf.m_pre_bkfit_filter)
      pre_filter = itconf.m_pre_bkfit_filter;
    else if (do_backward_fit)
      pre_filter = StdSeq::qfilter_nan_n_silly<TrackCand>;
    // pre_filter can be null if we are not doing backward fit as nan_n_silly will be run below.
    if (pre_filter)
      builder.filter_comb_cands(pre_filter, true);

    job.switch_to_backward();

    if (do_backward_fit) {
      if (itconf.m_backward_search) {
        builder.compactifyHitStorageForBestCand(itconf.m_backward_drop_seed_hits, itconf.m_backward_fit_min_hits);
      }

      builder.backwardFit();

      if (itconf.m_backward_search) {
        builder.beginBkwSearch();
        builder.findTracksCloneEngine(SteeringParams::IT_BkwSearch);
      }
    }

    // Post backward-fit filtering.
    filter_candidates_func post_filter;
    if (do_backward_fit && itconf.m_post_bkfit_filter)
      post_filter = [&](const TrackCand &tc, const MkJob &jb) -> bool {
        return itconf.m_post_bkfit_filter(tc, jb) && StdSeq::qfilter_nan_n_silly<TrackCand>(tc, jb);
      };
    else
      post_filter = StdSeq::qfilter_nan_n_silly<TrackCand>;
    // post_filter is always at least doing nan_n_silly filter.
    builder.filter_comb_cands(post_filter, true);

    if (do_backward_fit && itconf.m_backward_search)
      builder.endBkwSearch();

    builder.export_best_comb_cands(out_tracks, true);

    if (do_remove_duplicates && itconf.m_duplicate_cleaner) {
      itconf.m_duplicate_cleaner(out_tracks, itconf);
    }

    builder.end_event();
    builder.release_memory();
  }

}  // end namespace mkfit
