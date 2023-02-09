#include "RecoTracker/MkFitCMS/standalone/Shell.h"

#include "RecoTracker/MkFitCore/src/Debug.h"

// #include "RecoTracker/MkFitCore/src/Matriplex/MatriplexCommon.h"

#include "RecoTracker/MkFitCMS/interface/runFunctions.h"

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/MkBuilder.h"
#include "RecoTracker/MkFitCore/src/MkFitter.h"
#include "RecoTracker/MkFitCMS/interface/MkStdSeqs.h"
#include "RecoTracker/MkFitCMS/standalone/MkStandaloneSeqs.h"

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"

#include "RecoTracker/MkFitCore/standalone/Event.h"

#ifndef NO_ROOT
#include "TROOT.h"
#include "TRint.h"
#endif

#include "oneapi/tbb/task_arena.h"

#include <vector>

// clang-format off

namespace {
  constexpr int algos[] = {4, 22, 23, 5, 24, 7, 8, 9, 10, 6};  // 10 iterations
  constexpr int n_algos = sizeof(algos) / sizeof(int);

  const char* b2a(bool b) { return b ? "true" : "false"; }
}

namespace mkfit {

  Shell::Shell(std::vector<DeadVec> &dv, const std::string &in_file, int start_ev)
    : m_deadvectors(dv)
  {
    m_eoh = new EventOfHits(Config::TrkInfo);
    m_builder = new MkBuilder(Config::silent);

    m_backward_fit = Config::backwardFit;

    m_data_file = new DataFile;
    m_evs_in_file = m_data_file->openRead(in_file, Config::TrkInfo.n_layers());

    m_event = new Event(0, Config::TrkInfo.n_layers());
    GoToEvent(start_ev);
  }

  void Shell::Run() {
#ifndef NO_ROOT
    std::vector<const char *> argv = { "mkFit", "-l" };
    int argc = argv.size();
    TRint rint("mkFit-shell", &argc, const_cast<char**>(argv.data()));

    char buf[256];
    sprintf(buf, "mkfit::Shell &s = * (mkfit::Shell*) %p;", this);
    gROOT->ProcessLine(buf);
    printf("Shell &s variable is set\n");

    rint.Run(true);
    printf("Shell::Run finished\n");
#else
    printf("Shell::Run() no root, we rot -- erroring out. Recompile with WITH_ROOT=1\n");
    exit(1);
#endif
  }

  void Shell::Status() {
    printf("On event %d, selected iteration index %d, algo %d - %s\n"
          "  debug = %s, use_dead_modules = %s\n"
           "  clean_seeds = %s, backward_fit = %s, remove_duplicates = %s\n",
           m_event->evtID(), m_it_index, algos[m_it_index], TrackBase::algoint_to_cstr(algos[m_it_index]),
           b2a(g_debug), b2a(Config::useDeadModules),
           b2a(m_clean_seeds), b2a(m_backward_fit), b2a(m_remove_duplicates));
  }

  //===========================================================================
  // Event navigation / processing
  //===========================================================================

  void Shell::GoToEvent(int eid) {
    if (eid < 1) {
      fprintf(stderr, "Requested event %d is less than 1 -- 1 is the first event, %d is total number of events in file\n",
             eid, m_evs_in_file);
      throw std::runtime_error("event out of range");
    }
    if (eid > m_evs_in_file) {
      fprintf(stderr, "Requested event %d is grater than total number of events in file %d\n",
             eid, m_evs_in_file);
      throw std::runtime_error("event out of range");
    }

    int pos = m_event->evtID();
    if (eid > pos) {
      m_data_file->skipNEvents(eid - pos - 1);
    } else {
      m_data_file->rewind();
      m_data_file->skipNEvents(eid - 1);
    }
    m_event->reset(eid);
    m_event->read_in(*m_data_file);
    StdSeq::loadHitsAndBeamSpot(*m_event, *m_eoh);
    if (Config::useDeadModules) {
      StdSeq::loadDeads(*m_eoh, m_deadvectors);
    }

    printf("At event %d\n", eid);
  }

  void Shell::NextEvent(int skip) {
    GoToEvent(m_event->evtID() + skip);
  }

  void Shell::ProcessEvent(SeedSelect_e seed_select, int selected_seed, int count) {
    // count is only used for SS_IndexPreCleaning and SS_IndexPostCleaning.
    //       There are no checks for upper bounds, ie, if requested seeds beyond the first one exist.

    const IterationConfig &itconf = Config::ItrInfo[m_it_index];
    IterationMaskIfc mask_ifc;
    m_event->fill_hitmask_bool_vectors(itconf.m_track_algorithm, mask_ifc.m_mask_vector);

    m_seeds.clear();
    m_tracks.clear();

    {
      int n_algo = 0; // seeds are grouped by algo
      for (auto &s : m_event->seedTracks_) {
        if (s.algoint() == itconf.m_track_algorithm) {
          if (seed_select == SS_UseAll || seed_select == SS_IndexPostCleaning) {
            m_seeds.push_back(s);
          } else if (seed_select == SS_Label && s.label() == selected_seed) {
            m_seeds.push_back(s);
            break;
          } else if (seed_select == SS_IndexPreCleaning && n_algo >= selected_seed) {
            m_seeds.push_back(s);
            if (--count <= 0)
              break;
          }
          ++n_algo;
        } else if (n_algo > 0)
          break;
      }
    }

    printf("Shell::ProcessEvent running over %d seeds\n", (int) m_seeds.size());

    // Equivalent to run_OneIteration(...) without MkBuilder::release_memory().
    // If seed_select == SS_IndexPostCleaning the given seed is picked after cleaning.
    {
      const TrackerInfo &trackerInfo = Config::TrkInfo;
      const EventOfHits &eoh = *m_eoh;
      const IterationMaskIfcBase &it_mask_ifc = mask_ifc;
      MkBuilder &builder = *m_builder;
      TrackVec &seeds = m_seeds;
      TrackVec &out_tracks = m_tracks;
      bool do_seed_clean = m_clean_seeds;
      bool do_backward_fit = m_backward_fit;
      bool do_remove_duplicates = m_remove_duplicates;

      MkJob job({trackerInfo, itconf, eoh, eoh.refBeamSpot(), &it_mask_ifc});

      builder.begin_event(&job, m_event, __func__);

      // Seed cleaning not done on all iterations.
      do_seed_clean = m_clean_seeds && itconf.m_seed_cleaner;

      if (do_seed_clean)
        itconf.m_seed_cleaner(seeds, itconf, eoh.refBeamSpot());

      // Check nans in seeds -- this should not be needed when Slava fixes
      // the track parameter coordinate transformation.
      builder.seed_post_cleaning(seeds);

      if (seed_select == SS_IndexPostCleaning) {
        if (selected_seed >= 0 && selected_seed < (int)seeds.size()) {
          for (int i = 0; i < count; ++i)
            seeds[i] = seeds[selected_seed + i];
          seeds.resize(count);
        } else {
          seeds.clear();
        }
      }

      if (seeds.empty()) {
        if (seed_select != SS_UseAll)
          printf("Shell::ProcessEvent requested seed not found among seeds of the selected iteration.\n");
        else
          printf("Shell::ProcessEvent no seeds found.\n");
        return;
      }

      if (itconf.m_requires_seed_hit_sorting) {
        for (auto &s : seeds)
          s.sortHitsByLayer();  // sort seed hits for the matched hits (I hope it works here)
      }

      builder.find_tracks_load_seeds(seeds, do_seed_clean);

      builder.findTracksCloneEngine();

      printf("Shell::ProcessEvent post fwd search: %d comb-cands\n", builder.ref_eocc().size());

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

      printf("Shell::ProcessEvent post pre-bkf-filter (%s) and nan-filter (%s) filter: %d comb-cands\n",
             b2a(bool(itconf.m_pre_bkfit_filter)), b2a(do_backward_fit), builder.ref_eocc().size());

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

        printf("Shell::ProcessEvent post backward fit / search: %d comb-cands\n", builder.ref_eocc().size());
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

      printf("Shell::ProcessEvent post post-bkf-filter (%s) and nan-filter (true): %d comb-cands\n",
             b2a(do_backward_fit && itconf.m_post_bkfit_filter), builder.ref_eocc().size());

      if (do_backward_fit && itconf.m_backward_search)
        builder.endBkwSearch();

      builder.export_best_comb_cands(out_tracks, true);

      if (do_remove_duplicates && itconf.m_duplicate_cleaner) {
        itconf.m_duplicate_cleaner(out_tracks, itconf);
      }

      printf("Shell::ProcessEvent post remove-duplicates: %d comb-cands\n", (int) out_tracks.size());

      builder.end_event();
    }

    printf("Shell::ProcessEvent found %d tracks, number of seeds at end %d\n",
           (int) m_tracks.size(), (int) m_seeds.size());
  }

  //===========================================================================
  // Iteration selection
  //===========================================================================

  void Shell::SelectIterationIndex(int itidx) {
    if (itidx < 0 || itidx >= n_algos) {
      fprintf(stderr, "Requested iteration index out of range [%d, %d)", 0, n_algos);
      throw std::runtime_error("iteration index out of range");
    }
    m_it_index = itidx;
  }

  void Shell::SelectIterationAlgo(int algo) {
    for (int i = 0; i < n_algos; ++i) {
      if (algos[i] == algo) {
        m_it_index = i;
        return;
      }
    }
    fprintf(stderr, "Requested algo %d not found", algo);
    throw std::runtime_error("algo not found");
  }

  void Shell::PrintIterations() {
    printf("Shell::PrintIterations selected index = %d, %d iterations hardcoded as\n",
            m_it_index, n_algos);
    for (int i = 0; i < n_algos; ++i)
      printf("%d %2d %s\n", i, algos[i], TrackBase::algoint_to_cstr(algos[i]));
  }

  //===========================================================================
  // Flags / status setters
  //===========================================================================

  void Shell::SetDebug(bool b) { g_debug = b; }
  void Shell::SetCleanSeeds(bool b) { m_clean_seeds = b; }
  void Shell::SetBackwardFit(bool b) { m_backward_fit = b; }
  void Shell::SetRemoveDuplicates(bool b) { m_remove_duplicates = b; }
  void Shell::SetUseDeadModules(bool b) { Config::useDeadModules = b; }

  //===========================================================================
  // Analysis helpers
  //===========================================================================

  /*
    sim tracks are written to .bin files with a label equal to its own index.
    reco tracks labels are seed indices.
    seed labels are sim track indices
    --
    mkfit labels are seed indices in given iteration after cleaning (at seed load-time)
  */

  int Shell::LabelFromHits(Track &t, bool replace, float good_frac) {
    std::map<int, int> lab_cnt;
    for (int hi = 0; hi < t.nTotalHits(); ++hi) {
      auto hot = t.getHitOnTrack(hi);
      if (hot.index < 0)
        continue;
      const Hit &h = m_event->layerHits_[hot.layer][hot.index];
      int hl = m_event->simHitsInfo_[h.mcHitID()].mcTrackID_;
      if (hl >= 0)
        ++lab_cnt[hl];
    }
    int max_c = -1, max_l = -1;
    for (auto& x : lab_cnt) {
      if (x.second > max_c) {
        max_l = x.first;
        max_c = x.second;
      }
    }
    bool success = max_c >= good_frac * t.nFoundHits();
    int relabel = success ? max_l : -1;
    // printf("found_hits=%d, best_lab %d (%d hits), existing label=%d (replace flag=%s)\n",
    //        t.nFoundHits(), max_l, max_c, t.label(), b2a(replace));
    if (replace)
        t.setLabel(relabel);
    return relabel;
  }

  void Shell::FillByLabelMaps_CkfBase() {
    Event &ev = *m_event;
    const int track_algo = Config::ItrInfo[m_it_index].m_track_algorithm;

    m_ckf_map.clear();
    m_sim_map.clear();
    m_seed_map.clear();
    m_mkf_map.clear();

    // Pick ckf tracks with right algo and a good label.
    int rec_algo_match = 0;
    for (auto &t : ev.cmsswTracks_) {
      if (t.algoint() != track_algo)
        continue;
      ++rec_algo_match;
      int label = LabelFromHits(t, false, 0.5);
      if (label >= 0) {
        m_ckf_map.insert(std::make_pair(label, &t));
      }
    }

    // Pick sim tracks with labels found by ckf.
    for (auto &t : ev.simTracks_) {
      if (t.label() >= 0 && m_ckf_map.find(t.label()) != m_ckf_map.end()) {
        m_sim_map.insert(std::make_pair(t.label(), &t));
      }
    }

    // Pick seeds with right algo and a label found by ckf.
    for (auto &t : ev.seedTracks_) {
      if (t.algoint() == track_algo && t.label() >= 0 && m_ckf_map.find(t.label()) != m_ckf_map.end()) {
        m_seed_map.insert(std::make_pair(t.label(), &t));
      }
    }
    // Some seeds seem to be labeled -1, try fixing when not otherwise found.
    for (auto &t : ev.seedTracks_) {
      if (t.algoint() == track_algo && t.label() == -1) {
        int lab = LabelFromHits(t, true, 0.5);
        if (lab >= 0 && m_seed_map.find(lab) == m_seed_map.end()) {
          if (m_ckf_map.find(lab) != m_ckf_map.end()) {
            m_seed_map.insert(std::make_pair(t.label(), &t));
            printf("Saved seed with label -1 -> %d\n", lab);
          }
        }
      }
    }

    // Pick mkfit tracks, label by 
    for (auto &t : m_tracks) {
      int label = LabelFromHits(t, false, 0.5);
      if (label >= 0) {
        m_mkf_map.insert(std::make_pair(label, &t));
      }
    }

    printf("Shell::FillByLabelMaps reporting tracks with label >= 0, algo=%d (%s): "
           "ckf: %d of %d (same algo=%d)), sim: %d of %d, seed: %d of %d, mkfit: %d w/label of %d\n",
           track_algo, TrackBase::algoint_to_cstr(track_algo),
           (int) m_ckf_map.size(), (int) ev.cmsswTracks_.size(), rec_algo_match,
           (int) m_sim_map.size(), (int) ev.simTracks_.size(),
           (int) m_seed_map.size(), (int) m_seeds.size(),
           (int) m_mkf_map.size(), (int) m_tracks.size()
    );
  }

  bool Shell::CheckMkFitLayerPlanVsReferenceHits(const Track &mkft, const Track &reft, const std::string &name) {
    // Check if all hit-layers of a reference track reft are in the mkfit layer plan.
    // Returns true if all layers are in the plan.
    // String name is printed in front of label, expected to be SIMK or CKF.
    const IterationConfig &itconf = Config::ItrInfo[m_it_index];
    auto lp = itconf.m_steering_params[ mkft.getEtaRegion() ].m_layer_plan;
    bool ret = true;
    for (int hi = 0; hi < reft.nTotalHits(); ++hi) {
      auto hot = reft.getHitOnTrack(hi);
      if (std::find_if(lp.begin(), lp.end(), [=](auto &x){ return x.m_layer == hot.layer; }) == lp.end())
      {
        printf("CheckMkfLayerPlanVsCkfHits: layer %d not in layer plan for region %d, %s label=%d\n",
                hot.layer, mkft.getEtaRegion(), name.c_str(), reft.label());
        ret = false;
      }
    }
    return ret;
  }

  //===========================================================================
  // Analysis drivers / main functions / Comparators
  //===========================================================================

  void Shell::Compare() {
    Event &ev = *m_event;
    const IterationConfig &itconf = Config::ItrInfo[m_it_index];

    FillByLabelMaps_CkfBase();

    printf("------------------------------------------------------\n");

    const bool print_all_def = false;
    int mkf_cnt=0, less_hits=0, more_hits=0;

    // TOBTEC: look for rec-seeds with hits in tob1 and 2 only.
    int tot_cnt = 0, no_mkf_cnt = 0;

    for (auto& [l, simt_ptr]: m_sim_map)
    {
      auto &simt = * simt_ptr;
      auto &ckft = * m_ckf_map[l];
      auto mi = m_mkf_map.find(l);

      bool print_all = print_all_def;

      // TOBTEC: look for rec-seeds with hits in tob1 and 2 only.
      bool select = true;
      {
        auto &ckf_seed = ev.seedTracks_[ckft.label()];
        for (int hi = 0; hi < ckf_seed.nTotalHits(); ++hi) {
          const HitOnTrack hot = ckf_seed.getHitOnTrack(hi);
          if (hot.index >= 0 && (hot.layer < 10 || hot.layer > 13)) {
            select = false;
            break;
          }
        }
      }
      if ( ! select) continue;

      ++tot_cnt;
      //print_all = true;

      if (mi != m_mkf_map.end())
      {
        auto &mkft = * mi->second;
        mkf_cnt++;
        if (mkft.nFoundHits() < ckft.nFoundHits()) ++less_hits;
        if (mkft.nFoundHits() > ckft.nFoundHits()) ++more_hits;

        CheckMkFitLayerPlanVsReferenceHits(mkft, ckft, "CKF");
        // CheckMkFitLayerPlanVsReferenceHits(mkft, simt, "SIM");

        (void) print_all;
        if (/* itconf.m_track_algorithm == 10 ||*/ print_all) {
          // ckf label is wrong when validation is on (even quality val) for mixedTriplet, pixelless and tobtec
          // as seed tracks get removed for non-mkfit iterations and indices from rec-tracks are no longer valid.
          auto &ckf_seed = ev.seedTracks_[ckft.label()];
          auto &mkf_seed = m_seeds[mkft.label()];
          print("ckf  ", 0, ckft, ev);
          print("mkfit", 0, mkft, ev);
          print("sim  ", 0, simt, ev);
          print("ckf seed", 0, ckf_seed, ev);
          print("mkf seed", 0, mkf_seed, ev);
          printf("------------------------------------------------------\n");

          TrackVec ssss;
          ssss.push_back(mkf_seed);

          IterationSeedPartition pppp(1);
          IterationConfig::get_seed_partitioner("phase1:1:debug")(Config::TrkInfo, ssss, *m_eoh, pppp);

          printf("------------------------------------------------------\n");
          printf("\n");
        }
      }
      else
      {
        printf("\n!!!!! No mkfit track with this label.\n\n");
        ++no_mkf_cnt;

        auto &ckf_seed = ev.seedTracks_[ckft.label()];
        print("ckf ", 0, ckft, ev);
        print("sim ", 0, simt, ev);
        print("ckf seed", 0, ckf_seed, ev);
        auto smi = m_seed_map.find(l);
        if (smi != m_seed_map.end())
          print("seed with matching label", 0, *smi->second, ev);
        printf("------------------------------------------------------\n");
      }
    }

    printf("mkFit found %d, matching_label=%d, less_hits=%d, more_hits=%d  [algo=%d (%s)]\n",
           (int) ev.fitTracks_.size(), mkf_cnt, less_hits, more_hits,
           itconf.m_track_algorithm, TrackBase::algoint_to_cstr(itconf.m_track_algorithm));

    if (tot_cnt > 0) {
      printf("\ntobtec tob1/2 tot=%d no_mkf=%d (%f%%)\n",
            tot_cnt, no_mkf_cnt, 100.0 * no_mkf_cnt / tot_cnt);
    } else {
      printf("\nNo CKF tracks with seed hits in TOB1/2 found (need iteration idx 8, TobTec?)\n");
    }

    printf("-------------------------------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------------------------------\n");
    printf("\n");
  }

}
