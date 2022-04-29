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

    MkJob job({trackerInfo, itconf, eoh, &it_mask_ifc});

    builder.begin_event(&job, nullptr, __func__);

    if (do_seed_clean) {
      // Seed cleaning not done on pixelLess / tobTec iters
      if (itconf.m_requires_dupclean_tight)
        StdSeq::clean_cms_seedtracks_iter(&seeds, itconf, eoh.refBeamSpot());
    }

    // Check nans in seeds -- this should not be needed when Slava fixes
    // the track parameter coordinate transformation.
    builder.seed_post_cleaning(seeds);

    if (itconf.m_requires_seed_hit_sorting) {
      for (auto &s : seeds)
        s.sortHitsByLayer();  // sort seed hits for the matched hits (I hope it works here)
    }

    builder.find_tracks_load_seeds(seeds, do_seed_clean);

    builder.findTracksCloneEngine();

    using Algo = TrackBase::TrackAlgorithm;
    if (itconf.m_requires_quality_filter && Algo(itconf.m_track_algorithm) != Algo::detachedTripletStep) {
      if (Algo(itconf.m_track_algorithm) == Algo::pixelPairStep) {
        builder.filter_comb_cands([&](const TrackCand &t) { return StdSeq::qfilter_n_hits_pixseed(t, 3); });
      } else if (Algo(itconf.m_track_algorithm) == Algo::pixelLessStep) {
        builder.filter_comb_cands(
            [&](const TrackCand &t) { return StdSeq::qfilter_pixelLessFwd(t, eoh.refBeamSpot(), trackerInfo); });
      } else {
        builder.filter_comb_cands(
            [&](const TrackCand &t) { return StdSeq::qfilter_n_hits(t, itconf.m_params.minHitsQF); });
      }
    }

    if (do_backward_fit) {
      if (itconf.m_backward_search) {
        builder.compactifyHitStorageForBestCand(itconf.m_backward_drop_seed_hits, itconf.m_backward_fit_min_hits);
      }

      builder.backwardFit();

      if (itconf.m_backward_search) {
        builder.beginBkwSearch();
        builder.findTracksCloneEngine(SteeringParams::IT_BkwSearch);
        builder.endBkwSearch();
      }

      if (itconf.m_requires_quality_filter && (Algo(itconf.m_track_algorithm) == Algo::detachedTripletStep ||
                                               Algo(itconf.m_track_algorithm) == Algo::pixelLessStep)) {
        if (Algo(itconf.m_track_algorithm) == Algo::detachedTripletStep) {
          builder.filter_comb_cands(
              [&](const TrackCand &t) { return StdSeq::qfilter_n_layers(t, eoh.refBeamSpot(), trackerInfo); });
        } else if (Algo(itconf.m_track_algorithm) == Algo::pixelLessStep) {
          builder.filter_comb_cands(
              [&](const TrackCand &t) { return StdSeq::qfilter_pixelLessBkwd(t, eoh.refBeamSpot(), trackerInfo); });
        }
      }
    }

    builder.filter_comb_cands([&](const TrackCand &t) { return StdSeq::qfilter_nan_n_silly(t); });

    builder.export_best_comb_cands(out_tracks, true);

    if (do_remove_duplicates) {
      StdSeq::find_and_remove_duplicates(out_tracks, itconf);
    }

    builder.end_event();
    builder.release_memory();
  }

}  // end namespace mkfit
