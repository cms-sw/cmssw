#include <memory>
#include <limits>
#include <algorithm>

#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"

#include "RecoTracker/MkFitCore/interface/MkBuilder.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/binnor.h"

#include "Pool.h"
#include "CandCloner.h"
#include "FindingFoos.h"
#include "MkFitter.h"
#include "MkFinder.h"

#ifdef MKFIT_STANDALONE
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

//#define DEBUG
#include "Debug.h"
//#define DEBUG_FINAL_FIT

#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_for_each.h"

namespace mkfit {

  //==============================================================================
  // Execution context -- Pools of helper objects
  //==============================================================================

  struct ExecutionContext {
    ExecutionContext() = default;
    ~ExecutionContext() = default;

    Pool<CandCloner> m_cloners;
    Pool<MkFitter> m_fitters;
    Pool<MkFinder> m_finders;

    void populate(int n_thr) {
      m_cloners.populate(n_thr - m_cloners.size());
      m_fitters.populate(n_thr - m_fitters.size());
      m_finders.populate(n_thr - m_finders.size());
    }

    void clear() {
      m_cloners.clear();
      m_fitters.clear();
      m_finders.clear();
    }
  };

  CMS_SA_ALLOW ExecutionContext g_exe_ctx;

}  // end namespace mkfit

//------------------------------------------------------------------------------

namespace {
  using namespace mkfit;

  // Range of indices processed within one iteration of a TBB parallel_for.
  struct RangeOfSeedIndices {
    int m_rng_beg, m_rng_end;
    int m_beg, m_end;

    RangeOfSeedIndices(int rb, int re) : m_rng_beg(rb), m_rng_end(re) { reset(); }

    void reset() {
      m_end = m_rng_beg;
      next_chunk();
    }

    bool valid() const { return m_beg < m_rng_end; }

    int n_proc() const { return m_end - m_beg; }

    void next_chunk() {
      m_beg = m_end;
      m_end = std::min(m_end + NN, m_rng_end);
    }

    RangeOfSeedIndices &operator++() {
      next_chunk();
      return *this;
    }
  };

  // Region of seed indices processed in a single TBB parallel for.
  struct RegionOfSeedIndices {
    int m_reg_beg, m_reg_end, m_vec_cnt;

    RegionOfSeedIndices(const std::vector<int> &seedEtaSeparators, int region) {
      m_reg_beg = (region == 0) ? 0 : seedEtaSeparators[region - 1];
      m_reg_end = seedEtaSeparators[region];
      m_vec_cnt = (m_reg_end - m_reg_beg + NN - 1) / NN;
    }

    int count() const { return m_reg_end - m_reg_beg; }

    tbb::blocked_range<int> tbb_blk_rng_std(int thr_hint = -1) const {
      if (thr_hint < 0)
        thr_hint = Config::numSeedsPerTask;
      return tbb::blocked_range<int>(m_reg_beg, m_reg_end, thr_hint);
    }

    tbb::blocked_range<int> tbb_blk_rng_vec() const {
      return tbb::blocked_range<int>(0, m_vec_cnt, std::max(1, Config::numSeedsPerTask / NN));
    }

    RangeOfSeedIndices seed_rng(const tbb::blocked_range<int> &i) const {
      return RangeOfSeedIndices(m_reg_beg + NN * i.begin(), std::min(m_reg_beg + NN * i.end(), m_reg_end));
    }
  };

#ifdef DEBUG
  void pre_prop_print(int ilay, MkBase *fir) {
    const float pt = 1.f / fir->getPar(0, 0, 3);
    std::cout << "propagate to lay=" << ilay << " start from x=" << fir->getPar(0, 0, 0)
              << " y=" << fir->getPar(0, 0, 1) << " z=" << fir->getPar(0, 0, 2)
              << " r=" << getHypot(fir->getPar(0, 0, 0), fir->getPar(0, 0, 1))
              << " px=" << pt * std::cos(fir->getPar(0, 0, 4)) << " py=" << pt * std::sin(fir->getPar(0, 0, 4))
              << " pz=" << pt / std::tan(fir->getPar(0, 0, 5)) << " pT=" << pt << std::endl;
  }

  void post_prop_print(int ilay, MkBase *fir) {
    std::cout << "propagate to lay=" << ilay << " arrive at x=" << fir->getPar(0, 1, 0) << " y=" << fir->getPar(0, 1, 1)
              << " z=" << fir->getPar(0, 1, 2) << " r=" << getHypot(fir->getPar(0, 1, 0), fir->getPar(0, 1, 1))
              << std::endl;
  }

  void print_seed(const Track &seed) {
    std::cout << "MX - found seed with label=" << seed.label() << " nHits=" << seed.nFoundHits()
              << " chi2=" << seed.chi2() << " posEta=" << seed.posEta() << " posPhi=" << seed.posPhi()
              << " posR=" << seed.posR() << " posZ=" << seed.z() << " pT=" << seed.pT() << std::endl;
  }

  void print_seed2(const TrackCand &seed) {
    std::cout << "MX - found seed with nFoundHits=" << seed.nFoundHits() << " chi2=" << seed.chi2() << " x=" << seed.x()
              << " y=" << seed.y() << " z=" << seed.z() << " px=" << seed.px() << " py=" << seed.py()
              << " pz=" << seed.pz() << " pT=" << seed.pT() << std::endl;
  }

  void print_seeds(const TrackVec &seeds) {
    std::cout << "found total seeds=" << seeds.size() << std::endl;
    for (auto &&seed : seeds) {
      print_seed(seed);
    }
  }

  [[maybe_unused]] void print_seeds(const EventOfCombCandidates &event_of_comb_cands) {
    for (int iseed = 0; iseed < event_of_comb_cands.size(); iseed++) {
      print_seed2(event_of_comb_cands[iseed].front());
    }
  }
#endif

  bool sortCandByScore(const TrackCand &cand1, const TrackCand &cand2) {
    return mkfit::sortByScoreTrackCand(cand1, cand2);
  }

#ifdef RNT_DUMP_MkF_SelHitIdcs
  constexpr bool alwaysUseHitSelectionV2 = true;
#else
  constexpr bool alwaysUseHitSelectionV2 = false;
#endif
}  // end unnamed namespace

//------------------------------------------------------------------------------
// Constructor and destructor
//------------------------------------------------------------------------------

namespace mkfit {

  std::unique_ptr<MkBuilder> MkBuilder::make_builder(bool silent) { return std::make_unique<MkBuilder>(silent); }

  void MkBuilder::populate() { g_exe_ctx.populate(Config::numThreadsFinder); }
  void MkBuilder::clear() { g_exe_ctx.clear(); }

  std::pair<int, int> MkBuilder::max_hits_layer(const EventOfHits &eoh) const {
    int maxN = 0;
    int maxL = 0;
    for (int l = 0; l < eoh.nLayers(); ++l) {
      int lsize = eoh[l].nHits();
      if (lsize > maxN) {
        maxN = lsize;
        maxL = eoh[l].layer_id();
      }
    }
    return {maxN, maxL};
  }

  int MkBuilder::total_cands() const {
    int res = 0;
    for (int i = 0; i < m_event_of_comb_cands.size(); ++i)
      res += m_event_of_comb_cands[i].size();
    return res;
  }

  //------------------------------------------------------------------------------
  // Common functions
  //------------------------------------------------------------------------------

  void MkBuilder::begin_event(MkJob *job, Event *ev, const char *build_type) {
    m_nan_n_silly_per_layer_count = 0;

    m_job = job;
    m_event = ev;

    m_seedEtaSeparators.resize(m_job->num_regions());
    m_seedMinLastLayer.resize(m_job->num_regions());
    m_seedMaxLastLayer.resize(m_job->num_regions());

    for (int i = 0; i < m_job->num_regions(); ++i) {
      m_seedEtaSeparators[i] = 0;
      m_seedMinLastLayer[i] = 9999;
      m_seedMaxLastLayer[i] = 0;
    }

    if (!m_silent) {
      std::cout << "MkBuilder building tracks with '" << build_type << "'"
                << ", iteration_index=" << job->m_iter_config.m_iteration_index
                << ", track_algorithm=" << job->m_iter_config.m_track_algorithm << std::endl;
    }
  }

  void MkBuilder::end_event() {
    m_job = nullptr;
    m_event = nullptr;
  }

  void MkBuilder::release_memory() {
    TrackVec tmp;
    m_tracks.swap(tmp);
    m_event_of_comb_cands.releaseMemory();
  }

  void MkBuilder::import_seeds(const TrackVec &in_seeds,
                               const bool seeds_sorted,
                               std::function<insert_seed_foo> insert_seed) {
    // bool debug = true;

    const int size = in_seeds.size();

    IterationSeedPartition part(size);
    std::vector<unsigned> ranks;
    if (!seeds_sorted) {
      // We don't care about bins in phi, use low N to reduce overall number of bins.
      axis_pow2_u1<float, unsigned short, 10, 4> ax_phi(-Const::PI, Const::PI);
      axis<float, unsigned short, 8, 8> ax_eta(-3.0, 3.0, 64u);
      binnor<unsigned int, decltype(ax_phi), decltype(ax_eta), 20, 12> phi_eta_binnor(ax_phi, ax_eta);
      part.m_phi_eta_foo = [&](float phi, float eta) { phi_eta_binnor.register_entry_safe(phi, eta); };

      phi_eta_binnor.begin_registration(size);
      m_job->m_iter_config.m_seed_partitioner(m_job->m_trk_info, in_seeds, m_job->m_event_of_hits, part);
      phi_eta_binnor.finalize_registration();
      ranks.swap(phi_eta_binnor.m_ranks);
    } else {
      m_job->m_iter_config.m_seed_partitioner(m_job->m_trk_info, in_seeds, m_job->m_event_of_hits, part);
    }

    for (int i = 0; i < size; ++i) {
      int j = seeds_sorted ? i : ranks[i];
      int reg = part.m_region[j];
      ++m_seedEtaSeparators[reg];
    }

    // Sum up region counts to contain actual ending indices and prepare insertion cursors.
    // Fix min/max layers.
    std::vector<int> seed_cursors(m_job->num_regions());
    for (int reg = 1; reg < m_job->num_regions(); ++reg) {
      seed_cursors[reg] = m_seedEtaSeparators[reg - 1];
      m_seedEtaSeparators[reg] += m_seedEtaSeparators[reg - 1];
    }

    // Actually imports seeds, detect last-hit layer range per region.
    for (int i = 0; i < size; ++i) {
      int j = seeds_sorted ? i : ranks[i];
      int reg = part.m_region[j];
      const Track &seed = in_seeds[j];
      insert_seed(seed, j, reg, seed_cursors[reg]++);

      HitOnTrack hot = seed.getLastHitOnTrack();
      m_seedMinLastLayer[reg] = std::min(m_seedMinLastLayer[reg], hot.layer);
      m_seedMaxLastLayer[reg] = std::max(m_seedMaxLastLayer[reg], hot.layer);
    }

    // Fix min/max layers
    for (int reg = 0; reg < m_job->num_regions(); ++reg) {
      if (m_seedMinLastLayer[reg] == 9999)
        m_seedMinLastLayer[reg] = -1;
      if (m_seedMaxLastLayer[reg] == 0)
        m_seedMaxLastLayer[reg] = -1;
    }

    // clang-format off
    dprintf("MkBuilder::import_seeds finished import of %d seeds (last seeding layer min, max):\n"
            "  ec- = %d(%d,%d), t- = %d(%d,%d), brl = %d(%d,%d), t+ = %d(%d,%d), ec+ = %d(%d,%d).\n",
            size,
            m_seedEtaSeparators[0],                          m_seedMinLastLayer[0], m_seedMaxLastLayer[0],
            m_seedEtaSeparators[1] - m_seedEtaSeparators[0], m_seedMinLastLayer[1], m_seedMaxLastLayer[1],
            m_seedEtaSeparators[2] - m_seedEtaSeparators[1], m_seedMinLastLayer[2], m_seedMaxLastLayer[2],
            m_seedEtaSeparators[3] - m_seedEtaSeparators[2], m_seedMinLastLayer[3], m_seedMaxLastLayer[3],
            m_seedEtaSeparators[4] - m_seedEtaSeparators[3], m_seedMinLastLayer[4], m_seedMaxLastLayer[4]);
    // dcall(print_seeds(m_event_of_comb_cands));
    // clang-format on
  }

  //------------------------------------------------------------------------------

  int MkBuilder::filter_comb_cands(filter_candidates_func filter, bool attempt_all_cands) {
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    int i = 0, place_pos = 0;

    dprintf("MkBuilder::filter_comb_cands Entering filter size eoccs.size=%d\n", eoccs.size());

    std::vector<int> removed_cnts(m_job->num_regions());
    while (i < eoccs.size()) {
      if (eoccs.cands_in_backward_rep())
        eoccs[i].repackCandPostBkwSearch(0);
      bool passed = filter(eoccs[i].front(), *m_job);

      if (!passed && attempt_all_cands) {
        for (int j = 1; j < (int)eoccs[i].size(); ++j) {
          if (eoccs.cands_in_backward_rep())
            eoccs[i].repackCandPostBkwSearch(j);
          if (filter(eoccs[i][j], *m_job)) {
            eoccs[i][0] = eoccs[i][j];  // overwrite front, no need to std::swap() them
            passed = true;
            break;
          }
        }
      }
      if (passed) {
        if (place_pos != i)
          std::swap(eoccs[place_pos], eoccs[i]);
        ++place_pos;
      } else {
        assert(eoccs[i].front().getEtaRegion() < m_job->num_regions());
        ++removed_cnts[eoccs[i].front().getEtaRegion()];
      }
      ++i;
    }

    int n_removed = 0;
    for (int reg = 0; reg < m_job->num_regions(); ++reg) {
      dprintf("MkBuilder::filter_comb_cands reg=%d: n_rem_was=%d removed_in_r=%d n_rem=%d, es_was=%d es_new=%d\n",
              reg,
              n_removed,
              removed_cnts[reg],
              n_removed + removed_cnts[reg],
              m_seedEtaSeparators[reg],
              m_seedEtaSeparators[reg] - n_removed - removed_cnts[reg]);

      n_removed += removed_cnts[reg];
      m_seedEtaSeparators[reg] -= n_removed;
    }

    eoccs.resizeAfterFiltering(n_removed);

    dprintf("MkBuilder::filter_comb_cands n_removed = %d, eoccs.size=%d\n", n_removed, eoccs.size());

    return n_removed;
  }

  void MkBuilder::find_min_max_hots_size() {
    const EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    int min[5], max[5], gmin = 0, gmax = 0;
    int i = 0;
    for (int reg = 0; reg < 5; ++reg) {
      min[reg] = 9999;
      max[reg] = 0;
      for (; i < m_seedEtaSeparators[reg]; i++) {
        min[reg] = std::min(min[reg], eoccs[i].hotsSize());
        max[reg] = std::max(max[reg], eoccs[i].hotsSize());
      }
      gmin = std::max(gmin, min[reg]);
      gmax = std::max(gmax, max[reg]);
    }
    // clang-format off
    printf("MkBuilder::find_min_max_hots_size MIN %3d -- [ %3d | %3d | %3d | %3d | %3d ]   "
           "MAX %3d -- [ %3d | %3d | %3d | %3d | %3d ]\n",
           gmin, min[0], min[1], min[2], min[3], min[4],
           gmax, max[0], max[1], max[2], max[3], max[4]);
    // clang-format on
  }

  void MkBuilder::select_best_comb_cands(bool clear_m_tracks, bool remove_missing_hits) {
    if (clear_m_tracks)
      m_tracks.clear();
    export_best_comb_cands(m_tracks, remove_missing_hits);
  }

  void MkBuilder::export_best_comb_cands(TrackVec &out_vec, bool remove_missing_hits) {
    const EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    out_vec.reserve(out_vec.size() + eoccs.size());
    for (int i = 0; i < eoccs.size(); i++) {
      // Take the first candidate, if it exists.
      if (!eoccs[i].empty()) {
        const TrackCand &bcand = eoccs[i].front();
        out_vec.emplace_back(bcand.exportTrack(remove_missing_hits));
      }
    }
  }

  void MkBuilder::export_tracks(TrackVec &out_vec) {
    out_vec.reserve(out_vec.size() + m_tracks.size());
    for (auto &t : m_tracks) {
      out_vec.emplace_back(t);
    }
  }

  //------------------------------------------------------------------------------
  // PrepareSeeds
  //------------------------------------------------------------------------------

  void MkBuilder::seed_post_cleaning(TrackVec &tv) {
    if constexpr (Const::nan_n_silly_check_seeds) {
      int count = 0;

      for (int i = 0; i < (int)tv.size(); ++i) {
        bool silly = tv[i].hasSillyValues(Const::nan_n_silly_print_bad_seeds,
                                          Const::nan_n_silly_fixup_bad_seeds,
                                          "Post-cleaning seed silly value check and fix");
        if (silly) {
          ++count;
          if constexpr (Const::nan_n_silly_remove_bad_seeds) {
            // XXXX MT
            // Could do somethin smarter here: set as Stopped ?  check in seed cleaning ?
            tv.erase(tv.begin() + i);
            --i;
          }
        }
      }

      if (count > 0 && !m_silent) {
        printf("Nan'n'Silly detected %d silly seeds (fix=%d, remove=%d).\n",
               count,
               Const::nan_n_silly_fixup_bad_seeds,
               Const::nan_n_silly_remove_bad_seeds);
      }
    }
  }

  //------------------------------------------------------------------------------
  // FindTracksBestHit
  //------------------------------------------------------------------------------

  void MkBuilder::find_tracks_load_seeds_BH(const TrackVec &in_seeds, const bool seeds_sorted) {
    // bool debug = true;
    assert(!in_seeds.empty());
    m_tracks.resize(in_seeds.size());

    import_seeds(in_seeds, seeds_sorted, [&](const Track &seed, int seed_idx, int region, int pos) {
      m_tracks[pos] = seed;
      m_tracks[pos].setNSeedHits(seed.nTotalHits());
      m_tracks[pos].setEtaRegion(region);
    });

    //dump seeds
    dcall(print_seeds(m_tracks));
  }

  void MkBuilder::findTracksBestHit(SteeringParams::IterationType_e iteration_dir) {
    // bool debug = true;

    TrackVec &cands = m_tracks;

    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      if (iteration_dir == SteeringParams::IT_BkwSearch && !m_job->steering_params(region).has_bksearch_plan()) {
        printf("No backward search plan for region %d\n", region);
        return;
      }

      // XXXXXX Select endcap / barrel only ...
      // if (region != TrackerInfo::Reg_Endcap_Neg && region != TrackerInfo::Reg_Endcap_Pos)
      // if (region != TrackerInfo::Reg_Barrel)
      //   return;

      const SteeringParams &st_par = m_job->steering_params(region);
      const TrackerInfo &trk_info = m_job->m_trk_info;
      const PropagationConfig &prop_config = trk_info.prop_config();

      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      tbb::parallel_for(rosi.tbb_blk_rng_vec(), [&](const tbb::blocked_range<int> &blk_rng) {
        auto mkfndr = g_exe_ctx.m_finders.makeOrGet();

        RangeOfSeedIndices rng = rosi.seed_rng(blk_rng);

        std::vector<int> trk_idcs(NN);  // track indices in Matriplex
        std::vector<int> trk_llay(NN);  // last layer on input track

        while (rng.valid()) {
          dprint(std::endl << "processing track=" << rng.m_beg << ", label=" << cands[rng.m_beg].label());

          int prev_layer = 9999;

          for (int i = rng.m_beg, ii = 0; i < rng.m_end; ++i, ++ii) {
            int llay = cands[i].getLastHitLyr();
            trk_llay[ii] = llay;
            prev_layer = std::min(prev_layer, llay);

            dprintf("  %2d %2d %2d lay=%3d prev_layer=%d\n", ii, i, cands[i].label(), llay, prev_layer);
          }
          int curr_tridx = 0;

          auto layer_plan_it = st_par.make_iterator(iteration_dir);

          dprintf("Made iterator for %d, first layer=%d ... end layer=%d\n",
                  iteration_dir,
                  layer_plan_it.layer(),
                  layer_plan_it.last_layer());

          assert(layer_plan_it.is_pickup_only());

          int curr_layer = layer_plan_it.layer();

          mkfndr->m_Stopped.setVal(0);

          // Loop over layers, starting from after the seed.
          // Consider inverting loop order and make layer outer, need to
          // trade off hit prefetching with copy-out of candidates.
          while (++layer_plan_it) {
            prev_layer = curr_layer;
            curr_layer = layer_plan_it.layer();
            mkfndr->setup(prop_config,
                          m_job->m_iter_config,
                          m_job->m_iter_config.m_params,
                          m_job->m_iter_config.m_layer_configs[curr_layer],
                          st_par,
                          m_job->get_mask_for_layer(curr_layer),
                          m_event,
                          region,
                          m_job->m_in_fwd);

            const LayerOfHits &layer_of_hits = m_job->m_event_of_hits[curr_layer];
            const LayerInfo &layer_info = trk_info.layer(curr_layer);
            const FindingFoos &fnd_foos = FindingFoos::get_finding_foos(layer_info.is_barrel());
            dprint("at layer " << curr_layer << ", nHits in layer " << layer_of_hits.nHits());

            // Pick up seeds that become active on current layer -- unless already fully loaded.
            if (curr_tridx < rng.n_proc()) {
              int prev_tridx = curr_tridx;

              for (int i = rng.m_beg, ii = 0; i < rng.m_end; ++i, ++ii) {
                if (trk_llay[ii] == prev_layer)
                  trk_idcs[curr_tridx++] = i;
              }
              if (curr_tridx > prev_tridx) {
                dprintf("added %d seeds, started with %d\n", curr_tridx - prev_tridx, prev_tridx);

                mkfndr->inputTracksAndHitIdx(cands, trk_idcs, prev_tridx, curr_tridx, false, prev_tridx);
              }
            }

            if (layer_plan_it.is_pickup_only())
              continue;

            dcall(pre_prop_print(curr_layer, mkfndr.get()));

            mkfndr->clearFailFlag();
            (mkfndr.get()->*fnd_foos.m_propagate_foo)(
                layer_info.propagate_to(), curr_tridx, prop_config.finding_inter_layer_pflags);

            dcall(post_prop_print(curr_layer, mkfndr.get()));

            mkfndr->selectHitIndices(layer_of_hits, curr_tridx);

            // Stop low-pT tracks that can not reach the current barrel layer.
            if (layer_info.is_barrel()) {
              const float r_min_sqr = layer_info.rin() * layer_info.rin();
              for (int i = 0; i < curr_tridx; ++i) {
                if (!mkfndr->m_Stopped[i]) {
                  if (mkfndr->radiusSqr(i, MkBase::iP) < r_min_sqr) {
                    if (region == TrackerInfo::Reg_Barrel) {
                      mkfndr->m_Stopped[i] = 1;
                      mkfndr->outputTrackAndHitIdx(cands[rng.m_beg + i], i, false);
                    }
                    mkfndr->m_XWsrResult[i].m_wsr = WSR_Outside;
                    mkfndr->m_XHitSize[i] = 0;
                  }
                } else {  // make sure we don't add extra work for AddBestHit
                  mkfndr->m_XWsrResult[i].m_wsr = WSR_Outside;
                  mkfndr->m_XHitSize[i] = 0;
                }
              }
            }

            // make candidates with best hit
            dprint("make new candidates");

            mkfndr->addBestHit(layer_of_hits, curr_tridx, fnd_foos);

            // Stop tracks that have reached N_max_holes.
            for (int i = 0; i < curr_tridx; ++i) {
              if (!mkfndr->m_Stopped[i] && mkfndr->bestHitLastHoT(i).index == -2) {
                mkfndr->m_Stopped[i] = 1;
                mkfndr->outputTrackAndHitIdx(cands[rng.m_beg + i], i, false);
              }
            }

          }  // end of layer loop

          mkfndr->outputNonStoppedTracksAndHitIdx(cands, trk_idcs, 0, curr_tridx, false);

          ++rng;
        }  // end of loop over candidates in a tbb chunk

        mkfndr->release();
      });  // end parallel_for over candidates in a region
    });    // end of parallel_for_each over regions
  }

  //------------------------------------------------------------------------------
  // FindTracksCombinatorial: Standard TBB and CloneEngine TBB
  //------------------------------------------------------------------------------

  void MkBuilder::find_tracks_load_seeds(const TrackVec &in_seeds, const bool seeds_sorted) {
    // This will sort seeds according to iteration configuration.
    assert(!in_seeds.empty());
    m_tracks.clear();  // m_tracks can be used for BkFit.

    m_event_of_comb_cands.reset((int)in_seeds.size(), m_job->max_max_cands());

    import_seeds(in_seeds, seeds_sorted, [&](const Track &seed, int seed_idx, int region, int pos) {
      m_event_of_comb_cands.insertSeed(seed, seed_idx, m_job->steering_params(region).m_track_scorer, region, pos);
    });
  }

  int MkBuilder::find_tracks_unroll_candidates(std::vector<std::pair<int, int>> &seed_cand_vec,
                                               int start_seed,
                                               int end_seed,
                                               int layer,
                                               int prev_layer,
                                               bool pickup_only,
                                               SteeringParams::IterationType_e iteration_dir) {
    int silly_count = 0;

    seed_cand_vec.clear();

    auto &iter_params = (iteration_dir == SteeringParams::IT_BkwSearch) ? m_job->m_iter_config.m_backward_params
                                                                        : m_job->m_iter_config.m_params;

    for (int iseed = start_seed; iseed < end_seed; ++iseed) {
      CombCandidate &ccand = m_event_of_comb_cands[iseed];

      if (ccand.state() == CombCandidate::Dormant && ccand.pickupLayer() == prev_layer) {
        ccand.setState(CombCandidate::Finding);
      }
      if (!pickup_only && ccand.state() == CombCandidate::Finding) {
        bool active = false;
        for (int ic = 0; ic < (int)ccand.size(); ++ic) {
          if (ccand[ic].getLastHitIdx() != -2) {
            // Stop candidates with pT<X GeV
            if (ccand[ic].pT() < iter_params.minPtCut) {
              ccand[ic].addHitIdx(-2, layer, 0.0f);
              continue;
            }
            // Check if the candidate is close to it's max_r, pi/2 - 0.2 rad (11.5 deg)
            if (iteration_dir == SteeringParams::IT_FwdSearch && ccand[ic].pT() < 1.2) {
              const float dphi = std::abs(ccand[ic].posPhi() - ccand[ic].momPhi());
              if (ccand[ic].posRsq() > 625.f && dphi > 1.371f && dphi < 4.512f) {
                // printf("Stopping cand at r=%f, posPhi=%.1f momPhi=%.2f pt=%.2f emomEta=%.2f\n",
                //        ccand[ic].posR(), ccand[ic].posPhi(), ccand[ic].momPhi(), ccand[ic].pT(), ccand[ic].momEta());
                ccand[ic].addHitIdx(-2, layer, 0.0f);
                continue;
              }
            }

            active = true;
            seed_cand_vec.push_back(std::pair<int, int>(iseed, ic));
            ccand[ic].resetOverlaps();

            if constexpr (Const::nan_n_silly_check_cands_every_layer) {
              if (ccand[ic].hasSillyValues(Const::nan_n_silly_print_bad_cands_every_layer,
                                           Const::nan_n_silly_fixup_bad_cands_every_layer,
                                           "Per layer silly check"))
                ++silly_count;
            }
          }
        }
        if (!active) {
          ccand.setState(CombCandidate::Finished);
        }
      }
    }

    if constexpr (Const::nan_n_silly_check_cands_every_layer && silly_count > 0) {
      m_nan_n_silly_per_layer_count += silly_count;
    }

    return seed_cand_vec.size();
  }

  void MkBuilder::find_tracks_handle_missed_layers(MkFinder *mkfndr,
                                                   const LayerInfo &layer_info,
                                                   std::vector<std::vector<TrackCand>> &tmp_cands,
                                                   const std::vector<std::pair<int, int>> &seed_cand_idx,
                                                   const int region,
                                                   const int start_seed,
                                                   const int itrack,
                                                   const int end) {
    // XXXX-1 If I miss a layer, insert the original track into tmp_cands
    // AND do not do it in FindCandidates as the position can be badly
    // screwed by then. See comment there, too.
    // One could also do a pre-check ... so as not to use up a slot.

    // bool debug = true;

    for (int ti = itrack; ti < end; ++ti) {
      TrackCand &cand = m_event_of_comb_cands[seed_cand_idx[ti].first][seed_cand_idx[ti].second];
      WSR_Result &w = mkfndr->m_XWsrResult[ti - itrack];

      // Low pT tracks can miss a barrel layer ... and should be stopped
      dprintf("WSR Check label %d, seed %d, cand %d score %f -> wsr %d, in_gap %d\n",
              cand.label(),
              seed_cand_idx[ti].first,
              seed_cand_idx[ti].second,
              cand.score(),
              w.m_wsr,
              w.m_in_gap);

      if (w.m_wsr == WSR_Failed) {
        // Fake outside so it does not get processed in FindTracks BH/Std/CE.
        // [ Should add handling of WSR_Failed there, perhaps. ]
        w.m_wsr = WSR_Outside;

        if (layer_info.is_barrel()) {
          dprintf("Barrel cand propagation failed, got to r=%f ... layer is %f - %f\n",
                  mkfndr->radius(ti - itrack, MkBase::iP),
                  layer_info.rin(),
                  layer_info.rout());
          // In barrel region, create a stopped replica. In transition region keep the original copy
          // as there is still a chance to hit endcaps.
          tmp_cands[seed_cand_idx[ti].first - start_seed].push_back(cand);
          if (region == TrackerInfo::Reg_Barrel) {
            dprintf(" creating extra stopped held back candidate\n");
            tmp_cands[seed_cand_idx[ti].first - start_seed].back().addHitIdx(-2, layer_info.layer_id(), 0);
          }
        }
        // Never happens for endcap / propToZ
      } else if (w.m_wsr == WSR_Outside) {
        dprintf(" creating extra held back candidate\n");
        tmp_cands[seed_cand_idx[ti].first - start_seed].push_back(cand);
      } else if (w.m_wsr == WSR_Edge) {
        // Do nothing special here, this case is handled also in MkFinder:findTracks()
      }
    }
  }

  //------------------------------------------------------------------------------
  // FindTracksCombinatorial: Standard TBB
  //------------------------------------------------------------------------------

  void MkBuilder::findTracksStandard(SteeringParams::IterationType_e iteration_dir) {
    // debug = true;

    EventOfCombCandidates &eoccs = m_event_of_comb_cands;

    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      if (iteration_dir == SteeringParams::IT_BkwSearch && !m_job->steering_params(region).has_bksearch_plan()) {
        printf("No backward search plan for region %d\n", region);
        return;
      }

      const TrackerInfo &trk_info = m_job->m_trk_info;
      const SteeringParams &st_par = m_job->steering_params(region);
      const IterationParams &params = m_job->params();
      const PropagationConfig &prop_config = trk_info.prop_config();

      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      // adaptive seeds per task based on the total estimated amount of work to divide among all threads
      const int adaptiveSPT = std::clamp(
          Config::numThreadsEvents * eoccs.size() / Config::numThreadsFinder + 1, 4, Config::numSeedsPerTask);
      dprint("adaptiveSPT " << adaptiveSPT << " fill " << rosi.count() << "/" << eoccs.size() << " region " << region);

      // loop over seeds
      tbb::parallel_for(rosi.tbb_blk_rng_std(adaptiveSPT), [&](const tbb::blocked_range<int> &seeds) {
        auto mkfndr = g_exe_ctx.m_finders.makeOrGet();

        const int start_seed = seeds.begin();
        const int end_seed = seeds.end();
        const int n_seeds = end_seed - start_seed;

        std::vector<std::vector<TrackCand>> tmp_cands(n_seeds);
        for (size_t iseed = 0; iseed < tmp_cands.size(); ++iseed) {
          tmp_cands[iseed].reserve(2 * params.maxCandsPerSeed);  //factor 2 seems reasonable to start with
        }

        std::vector<std::pair<int, int>> seed_cand_idx;
        seed_cand_idx.reserve(n_seeds * params.maxCandsPerSeed);

        auto layer_plan_it = st_par.make_iterator(iteration_dir);

        dprintf("Made iterator for %d, first layer=%d ... end layer=%d\n",
                iteration_dir,
                layer_plan_it.layer(),
                layer_plan_it.last_layer());

        assert(layer_plan_it.is_pickup_only());

        int curr_layer = layer_plan_it.layer(), prev_layer;

        dprintf("\nMkBuilder::FindTracksStandard region=%d, seed_pickup_layer=%d, first_layer=%d\n",
                region,
                curr_layer,
                layer_plan_it.next_layer());

        auto &iter_params = (iteration_dir == SteeringParams::IT_BkwSearch) ? m_job->m_iter_config.m_backward_params
                                                                            : m_job->m_iter_config.m_params;

        // Loop over layers, starting from after the seed.
        while (++layer_plan_it) {
          prev_layer = curr_layer;
          curr_layer = layer_plan_it.layer();
          mkfndr->setup(prop_config,
                        m_job->m_iter_config,
                        iter_params,
                        m_job->m_iter_config.m_layer_configs[curr_layer],
                        st_par,
                        m_job->get_mask_for_layer(curr_layer),
                        m_event,
                        region,
                        m_job->m_in_fwd);

          const LayerOfHits &layer_of_hits = m_job->m_event_of_hits[curr_layer];
          const LayerInfo &layer_info = trk_info.layer(curr_layer);
          const FindingFoos &fnd_foos = FindingFoos::get_finding_foos(layer_info.is_barrel());

          dprintf("\n* Processing layer %d\n", curr_layer);
          mkfndr->begin_layer(layer_of_hits);

          int theEndCand = find_tracks_unroll_candidates(seed_cand_idx,
                                                         start_seed,
                                                         end_seed,
                                                         curr_layer,
                                                         prev_layer,
                                                         layer_plan_it.is_pickup_only(),
                                                         iteration_dir);

          dprintf("  Number of candidates to process: %d, nHits in layer: %d\n", theEndCand, layer_of_hits.nHits());

          if (layer_plan_it.is_pickup_only() || theEndCand == 0)
            continue;

          // vectorized loop
          for (int itrack = 0; itrack < theEndCand; itrack += NN) {
            int end = std::min(itrack + NN, theEndCand);

            dprint("processing track=" << itrack << ", label="
                                       << eoccs[seed_cand_idx[itrack].first][seed_cand_idx[itrack].second].label());

            //fixme find a way to deal only with the candidates needed in this thread
            mkfndr->inputTracksAndHitIdx(eoccs.refCandidates(), seed_cand_idx, itrack, end, false);

            //propagate to layer
            dcall(pre_prop_print(curr_layer, mkfndr.get()));

            mkfndr->clearFailFlag();
            (mkfndr.get()->*fnd_foos.m_propagate_foo)(
                layer_info.propagate_to(), end - itrack, prop_config.finding_inter_layer_pflags);

            dcall(post_prop_print(curr_layer, mkfndr.get()));

            dprint("now get hit range");

            if (alwaysUseHitSelectionV2 || iter_params.useHitSelectionV2)
              mkfndr->selectHitIndicesV2(layer_of_hits, end - itrack);
            else
              mkfndr->selectHitIndices(layer_of_hits, end - itrack);

            find_tracks_handle_missed_layers(
                mkfndr.get(), layer_info, tmp_cands, seed_cand_idx, region, start_seed, itrack, end);

            dprint("make new candidates");
            mkfndr->findCandidates(layer_of_hits, tmp_cands, start_seed, end - itrack, fnd_foos);

          }  //end of vectorized loop

          // sort the input candidates
          for (int is = 0; is < n_seeds; ++is) {
            dprint("dump seed n " << is << " with N_input_candidates=" << tmp_cands[is].size());

            std::sort(tmp_cands[is].begin(), tmp_cands[is].end(), sortCandByScore);
          }

          // now fill out the output candidates
          for (int is = 0; is < n_seeds; ++is) {
            if (!tmp_cands[is].empty()) {
              eoccs[start_seed + is].clear();

              // Put good candidates into eoccs, process -2 candidates.
              int n_placed = 0;
              bool first_short = true;
              for (int ii = 0; ii < (int)tmp_cands[is].size() && n_placed < params.maxCandsPerSeed; ++ii) {
                TrackCand &tc = tmp_cands[is][ii];

                // See if we have an overlap hit available, but only if we have a true hit in this layer
                // and pT is above the pTCutOverlap
                if (tc.pT() > params.pTCutOverlap && tc.getLastHitLyr() == curr_layer && tc.getLastHitIdx() >= 0) {
                  CombCandidate &ccand = eoccs[start_seed + is];

                  HitMatch *hm = ccand[tc.originIndex()].findOverlap(
                      tc.getLastHitIdx(), layer_of_hits.refHit(tc.getLastHitIdx()).detIDinLayer());

                  if (hm) {
                    tc.addHitIdx(hm->m_hit_idx, curr_layer, hm->m_chi2);
                    tc.incOverlapCount();
                  }
                }

                if (tc.getLastHitIdx() != -2) {
                  eoccs[start_seed + is].emplace_back(tc);
                  ++n_placed;
                } else if (first_short) {
                  first_short = false;
                  if (tc.score() > eoccs[start_seed + is].refBestShortCand().score()) {
                    eoccs[start_seed + is].setBestShortCand(tc);
                  }
                }
              }

              tmp_cands[is].clear();
            }
          }
          mkfndr->end_layer();
        }  // end of layer loop
        mkfndr->release();

        // final sorting
        for (int iseed = start_seed; iseed < end_seed; ++iseed) {
          eoccs[iseed].mergeCandsAndBestShortOne(m_job->params(), st_par.m_track_scorer, true, true);
        }
      });  // end parallel-for over chunk of seeds within region
    });    // end of parallel-for-each over eta regions

    // debug = false;
  }

  //------------------------------------------------------------------------------
  // FindTracksCombinatorial: CloneEngine TBB
  //------------------------------------------------------------------------------

  void MkBuilder::findTracksCloneEngine(SteeringParams::IterationType_e iteration_dir) {
    // debug = true;

    EventOfCombCandidates &eoccs = m_event_of_comb_cands;

    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      if (iteration_dir == SteeringParams::IT_BkwSearch && !m_job->steering_params(region).has_bksearch_plan()) {
        printf("No backward search plan for region %d\n", region);
        return;
      }

      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      // adaptive seeds per task based on the total estimated amount of work to divide among all threads
      const int adaptiveSPT = std::clamp(
          Config::numThreadsEvents * eoccs.size() / Config::numThreadsFinder + 1, 4, Config::numSeedsPerTask);
      dprint("adaptiveSPT " << adaptiveSPT << " fill " << rosi.count() << "/" << eoccs.size() << " region " << region);

      tbb::parallel_for(rosi.tbb_blk_rng_std(adaptiveSPT), [&](const tbb::blocked_range<int> &seeds) {
        auto cloner = g_exe_ctx.m_cloners.makeOrGet();
        auto mkfndr = g_exe_ctx.m_finders.makeOrGet();

        cloner->setup(m_job->params());

        // loop over layers
        find_tracks_in_layers(*cloner, mkfndr.get(), iteration_dir, seeds.begin(), seeds.end(), region);

        mkfndr->release();
        cloner->release();
      });
    });

    // debug = false;
  }

  void MkBuilder::find_tracks_in_layers(CandCloner &cloner,
                                        MkFinder *mkfndr,
                                        SteeringParams::IterationType_e iteration_dir,
                                        const int start_seed,
                                        const int end_seed,
                                        const int region) {
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    const TrackerInfo &trk_info = m_job->m_trk_info;
    const SteeringParams &st_par = m_job->steering_params(region);
    const IterationParams &params = m_job->params();
    const PropagationConfig &prop_config = trk_info.prop_config();

    const int n_seeds = end_seed - start_seed;

    std::vector<std::pair<int, int>> seed_cand_idx;
    std::vector<UpdateIndices> seed_cand_update_idx, seed_cand_overlap_idx;
    seed_cand_idx.reserve(n_seeds * params.maxCandsPerSeed);
    seed_cand_update_idx.reserve(n_seeds * params.maxCandsPerSeed);
    seed_cand_overlap_idx.reserve(n_seeds * params.maxCandsPerSeed);

    std::vector<std::vector<TrackCand>> extra_cands(n_seeds);
    for (int ii = 0; ii < n_seeds; ++ii)
      extra_cands[ii].reserve(params.maxCandsPerSeed);

    cloner.begin_eta_bin(&eoccs, &seed_cand_update_idx, &seed_cand_overlap_idx, &extra_cands, start_seed, n_seeds);

    // Loop over layers, starting from after the seed.

    auto layer_plan_it = st_par.make_iterator(iteration_dir);

    dprintf("Made iterator for %d, first layer=%d ... end layer=%d\n",
            iteration_dir,
            layer_plan_it.layer(),
            layer_plan_it.last_layer());

    assert(layer_plan_it.is_pickup_only());

    int curr_layer = layer_plan_it.layer(), prev_layer;

    dprintf(
        "\nMkBuilder::find_tracks_in_layers region=%d, seed_pickup_layer=%d, first_layer=%d; start_seed=%d, "
        "end_seed=%d\n",
        region,
        curr_layer,
        layer_plan_it.next_layer(),
        start_seed,
        end_seed);

    auto &iter_params = (iteration_dir == SteeringParams::IT_BkwSearch) ? m_job->m_iter_config.m_backward_params
                                                                        : m_job->m_iter_config.m_params;

    // Loop over layers according to plan.
    while (++layer_plan_it) {
      prev_layer = curr_layer;
      curr_layer = layer_plan_it.layer();
      mkfndr->setup(prop_config,
                    m_job->m_iter_config,
                    iter_params,
                    m_job->m_iter_config.m_layer_configs[curr_layer],
                    st_par,
                    m_job->get_mask_for_layer(curr_layer),
                    m_event,
                    region,
                    m_job->m_in_fwd);

      const bool pickup_only = layer_plan_it.is_pickup_only();

      const LayerInfo &layer_info = trk_info.layer(curr_layer);
      const LayerOfHits &layer_of_hits = m_job->m_event_of_hits[curr_layer];
      const FindingFoos &fnd_foos = FindingFoos::get_finding_foos(layer_info.is_barrel());

      dprintf("\n\n* Processing layer %d, %s\n\n", curr_layer, pickup_only ? "pickup only" : "full finding");
      mkfndr->begin_layer(layer_of_hits);

      const int theEndCand = find_tracks_unroll_candidates(
          seed_cand_idx, start_seed, end_seed, curr_layer, prev_layer, pickup_only, iteration_dir);

      dprintf("  Number of candidates to process: %d, nHits in layer: %d\n", theEndCand, layer_of_hits.nHits());

      // Don't bother messing with the clone engine if there are no candidates
      // (actually it crashes, so this protection is needed).
      // If there are no cands on this iteration, there won't be any later on either,
      // by the construction of the seed_cand_idx vector.
      // XXXXMT There might be cases in endcap where all tracks will miss the
      // next layer, but only relevant if we do geometric selection before.

      if (pickup_only || theEndCand == 0)
        continue;

      cloner.begin_layer(curr_layer);

      //vectorized loop
      for (int itrack = 0; itrack < theEndCand; itrack += NN) {
        const int end = std::min(itrack + NN, theEndCand);

#ifdef DEBUG
        dprintf("\nProcessing track=%d, start_seed=%d, n_seeds=%d, theEndCand=%d, end=%d, nn=%d, end_eq_tec=%d\n",
                itrack,
                start_seed,
                n_seeds,
                theEndCand,
                end,
                end - itrack,
                end == theEndCand);
        dprintf("  (seed,cand): ");
        for (int i = itrack; i < end; ++i)
          dprintf("(%d,%d)  ", seed_cand_idx[i].first, seed_cand_idx[i].second);
        dprintf("\n");
#endif

        mkfndr->inputTracksAndHitIdx(eoccs.refCandidates(), seed_cand_idx, itrack, end, false);

#ifdef DEBUG
        for (int i = itrack; i < end; ++i)
          dprintf("  track %d, idx %d is from seed %d\n", i, i - itrack, mkfndr->m_Label(i - itrack, 0, 0));
#endif

        // propagate to current layer
        mkfndr->clearFailFlag();
        (mkfndr->*fnd_foos.m_propagate_foo)(
            layer_info.propagate_to(), end - itrack, prop_config.finding_inter_layer_pflags);

        dprint("now get hit range");

        if (alwaysUseHitSelectionV2 || iter_params.useHitSelectionV2)
          mkfndr->selectHitIndicesV2(layer_of_hits, end - itrack);
        else
          mkfndr->selectHitIndices(layer_of_hits, end - itrack);

        find_tracks_handle_missed_layers(
            mkfndr, layer_info, extra_cands, seed_cand_idx, region, start_seed, itrack, end);

        // copy_out the propagated track params, errors only.
        // Do not, keep cands at last valid hit until actual update,
        // this requires change to propagation flags used in MkFinder::updateWithLastHit()
        // from intra-layer to inter-layer.
        // mkfndr->copyOutParErr(eoccs.refCandidates_nc(), end - itrack, true);

        // For prop-to-plane propagate from the last hit, not layer center.
        if constexpr (Config::usePropToPlane) {
          mkfndr->inputTracksAndHitIdx(eoccs.refCandidates(), seed_cand_idx, itrack, end, false);
        }

        dprint("make new candidates");
        cloner.begin_iteration();

        mkfndr->findCandidatesCloneEngine(layer_of_hits, cloner, start_seed, end - itrack, fnd_foos);

        cloner.end_iteration();
      }  //end of vectorized loop

      cloner.end_layer();

      // Update loop of best candidates. CandCloner prepares the list of those
      // that need update (excluding all those with negative last hit index).
      // This is split into two sections - candidates without overlaps and with overlaps.
      // On CMS PU-50 the ratio of those is ~ 65 : 35 over all iterations.
      // Note, overlap recheck is only enabled for some iterations, e.g. pixelLess.

      const int theEndUpdater = seed_cand_update_idx.size();

      for (int itrack = 0; itrack < theEndUpdater; itrack += NN) {
        const int end = std::min(itrack + NN, theEndUpdater);

        mkfndr->inputTracksAndHits(eoccs.refCandidates(), layer_of_hits, seed_cand_update_idx, itrack, end, true);

        mkfndr->updateWithLoadedHit(end - itrack, layer_of_hits, fnd_foos);

        // copy_out the updated track params, errors only (hit-idcs and chi2 already set)
        mkfndr->copyOutParErr(eoccs.refCandidates_nc(), end - itrack, false);
      }

      const int theEndOverlapper = seed_cand_overlap_idx.size();

      for (int itrack = 0; itrack < theEndOverlapper; itrack += NN) {
        const int end = std::min(itrack + NN, theEndOverlapper);

        mkfndr->inputTracksAndHits(eoccs.refCandidates(), layer_of_hits, seed_cand_overlap_idx, itrack, end, true);

        mkfndr->updateWithLoadedHit(end - itrack, layer_of_hits, fnd_foos);

        mkfndr->copyOutParErr(eoccs.refCandidates_nc(), end - itrack, false);

        mkfndr->inputOverlapHits(layer_of_hits, seed_cand_overlap_idx, itrack, end);

        // XXXX Could also be calcChi2AndUpdate(), then copy-out would have to be done
        // below, choosing appropriate slot (with or without the overlap hit).
        // Probably in a dedicated MkFinder copyOutXyzz function.
        mkfndr->chi2OfLoadedHit(end - itrack, fnd_foos);

        for (int ii = itrack; ii < end; ++ii) {
          const int fi = ii - itrack;
          TrackCand &tc = eoccs[seed_cand_overlap_idx[ii].seed_idx][seed_cand_overlap_idx[ii].cand_idx];

          // XXXX For now we DO NOT use chi2 as this was how things were done before the post-update
          // chi2 check. To use it we should retune scoring function (might be even simpler).
          auto chi2Ovlp = mkfndr->m_Chi2[fi];
          if (mkfndr->m_FailFlag[fi] == 0 && chi2Ovlp >= 0.0f && chi2Ovlp <= 60.0f) {
            auto scoreCand =
                getScoreCand(st_par.m_track_scorer, tc, true /*penalizeTailMissHits*/, true /*inFindCandidates*/);
            tc.addHitIdx(seed_cand_overlap_idx[ii].ovlp_idx, curr_layer, chi2Ovlp);
            tc.incOverlapCount();
            auto scoreCandOvlp = getScoreCand(st_par.m_track_scorer, tc, true, true);
            if (scoreCand > scoreCandOvlp)
              tc.popOverlap();
          }
        }
      }

      // Check if cands are sorted, as expected.
#ifdef DEBUG
      for (int iseed = start_seed; iseed < end_seed; ++iseed) {
        auto &cc = eoccs[iseed];

        for (int i = 0; i < ((int)cc.size()) - 1; ++i) {
          if (cc[i].score() < cc[i + 1].score()) {
            printf("CloneEngine - NOT SORTED: layer=%d, iseed=%d (size=%lu)-- %d : %f smaller than %d : %f\n",
                   curr_layer,
                   iseed,
                   cc.size(),
                   i,
                   cc[i].score(),
                   i + 1,
                   cc[i + 1].score());
          }
        }
      }
#endif
      mkfndr->end_layer();
    }  // end of layer loop

    cloner.end_eta_bin();

    // final sorting
    for (int iseed = start_seed; iseed < end_seed; ++iseed) {
      eoccs[iseed].mergeCandsAndBestShortOne(m_job->params(), st_par.m_track_scorer, true, true);
    }
  }

  //==============================================================================
  // BackwardFit
  //==============================================================================

#ifdef DEBUG_FINAL_FIT
  namespace {
    // clang-format off
    void dprint_tcand(const TrackCand &t, int i) {
      dprintf("  %4d with q=%+d chi2=%7.3f pT=%7.3f eta=% 7.3f x=%.3f y=%.3f z=%.3f"
              " nHits=%2d  label=%4d findable=%d\n",
              i, t.charge(), t.chi2(), t.pT(), t.momEta(), t.x(), t.y(), t.z(),
              t.nFoundHits(), t.label(), t.isFindable());
      }
    // clang-format on
  }  // namespace
#endif

  void MkBuilder::backwardFitBH() {
    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      tbb::parallel_for(rosi.tbb_blk_rng_vec(), [&](const tbb::blocked_range<int> &blk_rng) {
        auto mkfndr = g_exe_ctx.m_finders.makeOrGet();

        RangeOfSeedIndices rng = rosi.seed_rng(blk_rng);

        while (rng.valid()) {
          // final backward fit
          fit_cands_BH(mkfndr.get(), rng.m_beg, rng.m_end, region);

          ++rng;
        }
      });
    });
  }

  void MkBuilder::fit_cands_BH(MkFinder *mkfndr, int start_cand, int end_cand, int region) {
    const SteeringParams &st_par = m_job->steering_params(region);
    const PropagationConfig &prop_config = m_job->m_trk_info.prop_config();
    mkfndr->setup_bkfit(prop_config, st_par, m_event);
#ifdef DEBUG_FINAL_FIT
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    bool debug = true;
#endif

    for (int icand = start_cand; icand < end_cand; icand += NN) {
      const int end = std::min(icand + NN, end_cand);

#ifdef DEBUG_FINAL_FIT
      dprintf("Pre Final fit for %d - %d\n", icand, end);
      for (int i = icand; i < end; ++i) {
        dprint_tcand(eoccs[i][0], i);
      }
#endif

      bool chi_debug = false;
#ifdef DEBUG_BACKWARD_FIT_BH
    redo_fit:
#endif

      // input candidate tracks
      mkfndr->bkFitInputTracks(m_tracks, icand, end);

      // perform fit back to first layer on track
      mkfndr->bkFitFitTracksBH(m_job->m_event_of_hits, st_par, end - icand, chi_debug);

      // now move one last time to PCA
      if (prop_config.backward_fit_to_pca) {
        mkfndr->bkFitPropTracksToPCA(end - icand);
      }

#ifdef DEBUG_BACKWARD_FIT_BH
      // Dump tracks with pT > 2 and chi2/dof > 20. Assumes MPT_SIZE=1.
      if (!chi_debug && 1.0f / mkfndr->m_Par[MkBase::iP].At(0, 3, 0) > 2.0f &&
          mkfndr->m_Chi2(0, 0, 0) / (eoccs[icand][0].nFoundHits() * 3 - 6) > 20.0f) {
        chi_debug = true;
#ifdef MKFIT_STANDALONE
        printf("CHIHDR Event %d, Cand %3d, pT %f, chipdof %f ### NOTE x,y,z in cm, sigmas, deltas in mum ### !!!\n",
               m_event->evtID(),
#else
        printf("CHIHDR Cand %3d, pT %f, chipdof %f ### NOTE x,y,z in cm, sigmas, deltas in mum ### !!!\n",
#endif
               icand,
               1.0f / mkfndr->m_Par[MkBase::iP].At(0, 3, 0),
               mkfndr->m_Chi2(0, 0, 0) / (eoccs[icand][0].nFoundHits() * 3 - 6));
        // clang-format off
        printf("CHIHDR %3s %10s"
              " %10s %10s %10s %10s %11s %11s %11s"
              " %10s %10s %10s %10s %11s %11s %11s"
              " %10s %10s %10s %10s %10s %11s %11s\n",
              "lyr", "chi2",
              "x_h", "y_h", "z_h", "r_h", "sx_h", "sy_h", "sz_h",
              "x_t", "y_t", "z_t", "r_t", "sx_t", "sy_t", "sz_t",
              "pt", "phi", "theta", "phi_h", "phi_t", "d_xy", "d_z");
        // clang-format on
        goto redo_fit;
      }
#endif

      // copy out full set of info at last propagated position
      mkfndr->bkFitOutputTracks(m_tracks, icand, end, prop_config.backward_fit_to_pca);

#ifdef DEBUG_FINAL_FIT
      dprintf("Post Final fit for %d - %d\n", icand, end);
      for (int i = icand; i < end; ++i) {
        dprint_tcand(eoccs[i][0], i);
      }
#endif
    }
  }

  //------------------------------------------------------------------------------

  void MkBuilder::backwardFit() {
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;

    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      // adaptive seeds per task based on the total estimated amount of work to divide among all threads
      const int adaptiveSPT = std::clamp(
          Config::numThreadsEvents * eoccs.size() / Config::numThreadsFinder + 1, 4, Config::numSeedsPerTask);
      dprint("adaptiveSPT " << adaptiveSPT << " fill " << rosi.count() << "/" << eoccs.size() << " region " << region);

      tbb::parallel_for(rosi.tbb_blk_rng_std(adaptiveSPT), [&](const tbb::blocked_range<int> &cands) {
        auto mkfndr = g_exe_ctx.m_finders.makeOrGet();

        fit_cands(mkfndr.get(), cands.begin(), cands.end(), region);
      });
    });
  }

  void MkBuilder::fit_cands(MkFinder *mkfndr, int start_cand, int end_cand, int region) {
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    const SteeringParams &st_par = m_job->steering_params(region);
    const PropagationConfig &prop_config = m_job->m_trk_info.prop_config();
    mkfndr->setup_bkfit(prop_config, st_par, m_event);

    int step = NN;
    for (int icand = start_cand; icand < end_cand; icand += step) {
      int end = std::min(icand + NN, end_cand);

      bool chi_debug = false;

#ifdef DEBUG_FINAL_FIT
      bool debug = true;
      dprintf("Pre Final fit for %d - %d\n", icand, end);
      for (int i = icand; i < end; ++i) {
        dprint_tcand(eoccs[i][0], i);
      }
      chi_debug = true;
      static bool first = true;
      if (first) {
        // ./mkFit ... | perl -ne 'if (/^BKF_OVERLAP/) { s/^BKF_OVERLAP //og; print; }' > bkf_ovlp.rtt
        dprintf(
            "BKF_OVERLAP event/I:label/I:prod_type/I:is_findable/I:layer/I:is_stereo/I:is_barrel/I:"
            "pt/F:pt_cur/F:eta/F:phi/F:phi_cur/F:r_cur/F:z_cur/F:chi2/F:isnan/I:isfin/I:gtzero/I:hit_label/I:"
            "sx_t/F:sy_t/F:sz_t/F:d_xy/F:d_z/F\n");
        first = false;
      }
#endif

      // input tracks
      mkfndr->bkFitInputTracks(eoccs, icand, end);

      // fit tracks back to first layer
      mkfndr->bkFitFitTracks(m_job->m_event_of_hits, st_par, end - icand, chi_debug);

      // now move one last time to PCA
      if (prop_config.backward_fit_to_pca) {
        mkfndr->bkFitPropTracksToPCA(end - icand);
      }

      mkfndr->bkFitOutputTracks(eoccs, icand, end, prop_config.backward_fit_to_pca);

#ifdef DEBUG_FINAL_FIT
      dprintf("Post Final fit for %d - %d\n", icand, end);
      for (int i = icand; i < end; ++i) {
        dprint_tcand(eoccs[i][0], i);
      }
#endif
    }
    mkfndr->release();
  }

}  // end namespace mkfit
