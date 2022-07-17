#include "CandCloner.h"

#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

//#define DEBUG
#include "Debug.h"

namespace {
  inline bool sortCandListByScore(const mkfit::IdxChi2List &cand1, const mkfit::IdxChi2List &cand2) {
    return mkfit::sortByScoreStruct(cand1, cand2);
  }
}  // namespace

namespace mkfit {

  void CandCloner::setup(const IterationParams &ip) {
    mp_iteration_params = &ip;
    for (int iseed = 0; iseed < s_max_seed_range; ++iseed) {
      t_cands_for_next_lay[iseed].reserve(mp_iteration_params->maxCandsPerSeed);
    }
  }

  void CandCloner::release() { mp_iteration_params = nullptr; }

  void CandCloner::begin_eta_bin(EventOfCombCandidates *e_o_ccs,
                                 std::vector<std::pair<int, int>> *update_list,
                                 std::vector<std::vector<TrackCand>> *extra_cands,
                                 int start_seed,
                                 int n_seeds) {
    mp_event_of_comb_candidates = e_o_ccs;
    mp_kalman_update_list = update_list;
    mp_extra_cands = extra_cands;
    m_start_seed = start_seed;
    m_n_seeds = n_seeds;
    m_hits_to_add.resize(n_seeds);

    for (int i = 0; i < n_seeds; ++i)
      m_hits_to_add[i].reserve(4);

#ifdef CC_TIME_ETA
    printf("CandCloner::begin_eta_bin\n");
    t_eta = dtime();
#endif
  }

  void CandCloner::begin_layer(int lay) {
    m_layer = lay;

    m_idx_max = 0;
    m_idx_max_prev = 0;

    mp_kalman_update_list->clear();

#ifdef CC_TIME_LAYER
    t_lay = dtime();
#endif
  }

  void CandCloner::begin_iteration() {
    // Do nothing, "secondary" state vars updated when work completed/assigned.
  }

  void CandCloner::end_iteration() {
    int proc_n = m_idx_max - m_idx_max_prev;

    dprintf("CandCloner::end_iteration process %d, max_prev=%d, max=%d\n", proc_n, m_idx_max_prev, m_idx_max);

    if (proc_n >= s_max_seed_range) {
      // Round to multiple of s_max_seed_range.
      doWork((m_idx_max / s_max_seed_range) * s_max_seed_range);
    }
  }

  void CandCloner::end_layer() {
    if (m_n_seeds > m_idx_max_prev) {
      doWork(m_n_seeds);
    }

    for (int i = 0; i < m_n_seeds; ++i) {
      m_hits_to_add[i].clear();
    }

#ifdef CC_TIME_LAYER
    t_lay = dtime() - t_lay;
    printf("CandCloner::end_layer %d -- t_lay=%8.6f\n", m_layer, t_lay);
    printf("                      m_idx_max=%d, m_idx_max_prev=%d, issued work=%d\n",
           m_idx_max,
           m_idx_max_prev,
           m_idx_max + 1 > m_idx_max_prev);
#endif
  }

  void CandCloner::end_eta_bin() {
#ifdef CC_TIME_ETA
    t_eta = dtime() - t_eta;
    printf("CandCloner::end_eta_bin t_eta=%8.6f\n", t_eta);
#endif
  }
  //==============================================================================

  void CandCloner::doWork(int idx) {
    dprintf("CandCloner::DoWork assigning work from seed %d to %d\n", m_idx_max_prev, idx);

    int beg = m_idx_max_prev;
    int the_end = idx;

    dprintf("CandCloner::DoWork working on beg=%d to the_end=%d\n", beg, the_end);

    while (beg != the_end) {
      int end = std::min(beg + s_max_seed_range, the_end);

      dprintf("CandCloner::DoWork processing %4d -> %4d\n", beg, end);

      processSeedRange(beg, end);

      beg = end;
    }

    m_idx_max_prev = idx;
  }

  //==============================================================================

  void CandCloner::processSeedRange(int is_beg, int is_end) {
    // Process new hits for a range of seeds.

    // bool debug = true;

    dprintf("\nCandCloner::ProcessSeedRange is_beg=%d, is_end=%d\n", is_beg, is_end);

    //1) sort the candidates
    for (int is = is_beg; is < is_end; ++is) {
      std::vector<IdxChi2List> &hitsForSeed = m_hits_to_add[is];

      CombCandidate &ccand = mp_event_of_comb_candidates->cand(m_start_seed + is);
      std::vector<TrackCand> &extras = (*mp_extra_cands)[is];
      auto extra_i = extras.begin();
      auto extra_e = extras.end();

      // Extras are sorted by candScore.

#ifdef DEBUG
      dprint("  seed n " << is << " with input candidates=" << hitsForSeed.size());
      for (int ih = 0; ih < (int)hitsForSeed.size(); ih++) {
        dprint("trkIdx=" << hitsForSeed[ih].trkIdx << " hitIdx=" << hitsForSeed[ih].hitIdx
                         << " chi2=" << hitsForSeed[ih].chi2 << std::endl
                         << "    "
                         << "original pt=" << ccand[hitsForSeed[ih].trkIdx].pT() << " "
                         << "nTotalHits=" << ccand[hitsForSeed[ih].trkIdx].nTotalHits() << " "
                         << "nFoundHits=" << ccand[hitsForSeed[ih].trkIdx].nFoundHits() << " "
                         << "chi2=" << ccand[hitsForSeed[ih].trkIdx].chi2());
      }
#endif

      if (!hitsForSeed.empty()) {
        //sort the new hits
        std::sort(hitsForSeed.begin(), hitsForSeed.end(), sortCandListByScore);

        int num_hits = std::min((int)hitsForSeed.size(), mp_iteration_params->maxCandsPerSeed);

        // This is from buffer, we know it was cleared after last usage.
        std::vector<TrackCand> &cv = t_cands_for_next_lay[is - is_beg];

        int n_pushed = 0;

        for (int ih = 0; ih < num_hits; ih++) {
          const IdxChi2List &h2a = hitsForSeed[ih];

          TrackCand tc(ccand[h2a.trkIdx]);
          tc.addHitIdx(h2a.hitIdx, m_layer, h2a.chi2_hit);
          tc.setScore(h2a.score);

          if (h2a.hitIdx == -2) {
            if (h2a.score > ccand.refBestShortCand().score()) {
              ccand.setBestShortCand(tc);
            }
            continue;
          }

          // Could also skip storing of cands with last -3 hit.

          // Squeeze in extra tracks that are better than current one.
          while (extra_i != extra_e && sortByScoreTrackCand(*extra_i, tc) &&
                 n_pushed < mp_iteration_params->maxCandsPerSeed) {
            cv.emplace_back(*extra_i);
            ++n_pushed;
            ++extra_i;
          }

          if (n_pushed >= mp_iteration_params->maxCandsPerSeed)
            break;

          // set the overlap if we have a true hit and pT > pTCutOverlap
          HitMatch *hm;
          if (tc.pT() > mp_iteration_params->pTCutOverlap && h2a.hitIdx >= 0 &&
              (hm = ccand[h2a.trkIdx].findOverlap(h2a.hitIdx, h2a.module))) {
            tc.addHitIdx(hm->m_hit_idx, m_layer, hm->m_chi2);
            tc.incOverlapCount();
          }

          cv.emplace_back(tc);
          ++n_pushed;

          if (h2a.hitIdx >= 0) {
            mp_kalman_update_list->push_back(std::pair<int, int>(m_start_seed + is, n_pushed - 1));
          }
        }

        // Add remaining extras as long as there is still room for them.
        while (extra_i != extra_e && n_pushed < mp_iteration_params->maxCandsPerSeed) {
          cv.emplace_back(*extra_i);
          ++n_pushed;
          ++extra_i;
        }

        // Can not use ccand.swap(cv) -- allocations for TrackCand vectors need to be
        // in the same memory segment for gather operation to work in backward-fit.
        ccand.resize(cv.size());
        for (size_t ii = 0; ii < cv.size(); ++ii) {
          ccand[ii] = cv[ii];
        }
        cv.clear();
      } else  // hitsForSeed.empty()
      {
        if (ccand.state() == CombCandidate::Finding) {
          ccand.clear();

          while (extra_i != extra_e) {
            ccand.emplace_back(*extra_i);
            ++extra_i;
          }
        }
      }

      extras.clear();
    }
  }

}  // end namespace mkfit
