#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "Matriplex/Memory.h"

#include "Debug.h"

namespace mkfit {

  //==============================================================================
  // TrackCand
  //==============================================================================

  Track TrackCand::exportTrack(bool remove_missing_hits) const {
    dprintf("TrackCand::exportTrack label=%5d, total_hits=%2d, overlaps=%2d -- n_seed_hits=%d,prod_type=%d\n",
            label(),
            nTotalHits(),
            nOverlapHits_,
            getNSeedHits(),
            (int)prodType());

    Track res(*this);
    res.resizeHits(remove_missing_hits ? nFoundHits() : nTotalHits(), nFoundHits());
    res.setNOverlapHits(nOverlapHits());

    int nh = nTotalHits();
    int ch = lastHitIdx_;
    int good_hits_pos = nFoundHits();
    while (--nh >= 0) {
      const HoTNode &hot_node = m_comb_candidate->hot_node(ch);
      if (remove_missing_hits) {
        if (hot_node.m_hot.index >= 0)
          res.setHitIdxAtPos(--good_hits_pos, hot_node.m_hot);
      } else {
        res.setHitIdxAtPos(nh, hot_node.m_hot);
      }
      dprintf("  nh=%2d, ch=%d, idx=%d lyr=%d prev_idx=%d\n",
              nh,
              ch,
              hot_node.m_hot.index,
              hot_node.m_hot.layer,
              hot_node.m_prev_idx);
      ch = hot_node.m_prev_idx;
    }

    return res;
  }

  //==============================================================================
  // CombCandidate
  //==============================================================================

  void CombCandidate::importSeed(const Track &seed, const track_score_func &score_func, int region) {
    m_trk_cands.emplace_back(TrackCand(seed, this));

    m_state = CombCandidate::Dormant;
    m_pickup_layer = seed.getLastHitLyr();
#ifdef DUMPHITWINDOW
    m_seed_algo = seed.algoint();
    m_seed_label = seed.label();
#endif

    TrackCand &cand = m_trk_cands.back();
    cand.setNSeedHits(seed.nTotalHits());
    cand.setEtaRegion(region);

    dprintf("Importing pt=%f eta=%f, lastCcIndex=%d\n", cand.pT(), cand.momEta(), cand.lastCcIndex());

    for (const HitOnTrack *hp = seed.beginHitsOnTrack(); hp != seed.endHitsOnTrack(); ++hp) {
      dprintf(" hit idx=%d lyr=%d\n", hp->index, hp->layer);
      cand.addHitIdx(hp->index, hp->layer, 0.0f);
    }

    cand.setScore(getScoreCand(score_func, cand));
  }

  void CombCandidate::mergeCandsAndBestShortOne(const IterationParams &params,
                                                const track_score_func &score_func,
                                                bool update_score,
                                                bool sort_cands) {
    TrackCand *best_short = m_best_short_cand.combCandidate() ? &m_best_short_cand : nullptr;

    if (!empty()) {
      if (update_score) {
        for (auto &c : m_trk_cands)
          c.setScore(getScoreCand(score_func, c));
        if (best_short)
          best_short->setScore(getScoreCand(score_func, *best_short));
      }
      if (sort_cands) {
        std::sort(m_trk_cands.begin(), m_trk_cands.end(), sortByScoreTrackCand);
      }

      if (best_short && best_short->score() > m_trk_cands.back().score()) {
        auto ci = m_trk_cands.begin();
        while (ci->score() > best_short->score())
          ++ci;

        if ((int)m_trk_cands.size() >= params.maxCandsPerSeed)
          m_trk_cands.pop_back();

          // To print out what has been replaced -- remove when done with short track handling.
#ifdef DEBUG
        if (ci == m_trk_cands.begin()) {
          printf("FindTracksStd -- Replacing best cand (%f) with short one (%f) in final sorting\n",
                 m_trk_cands.front().score(),
                 best_short->score());
        }
#endif

        m_trk_cands.insert(ci, *best_short);
      }

    } else if (best_short) {
      m_trk_cands.push_back(*best_short);
    }

    if (best_short)
      best_short->resetShortTrack();

    // assert(capacity() == (size_t)Config::maxCandsPerSeed);
  }

  void CombCandidate::compactifyHitStorageForBestCand(bool remove_seed_hits, int backward_fit_min_hits) {
    // The best candidate is assumed to be in position 0 (after mergeCandsAndBestShortOne
    // mergeCandsAndBestShortOne has been called).
    // Other cands are dropped, their hits are dropped as well.
    // Seed hits are dropped if remove_seed_hits is true.

    /* The following considerations are related to the following implementation:
  minNrOfHitsForRebuild (checked against "nHits - nseed") has a default at 5, except
  1 in initialStep
  4 in tobTec and pixelLess
  https://github.com/cms-sw/cmssw/blob/master/RecoTracker/CkfPattern/plugins/GroupedCkfTrajectoryBuilder.cc#L1015

  NOTE: some of those can be matched hits !!!

  the hit splitting is triggered here: https://github.com/cms-sw/cmssw/blob/master/RecoTracker/CkfPattern/src/CkfTrackCandidateMakerBase.cc#L468
  after the rebuild has already happened: https://github.com/cms-sw/cmssw/blob/master/RecoTracker/CkfPattern/src/CkfTrackCandidateMakerBase.cc#L313
  */

    assert(!m_trk_cands.empty());
    m_trk_cands.resize(1);
    TrackCand &tc = m_trk_cands[0];

    // Do NOT remove any seed hits if fewer than backward_fit_min_hits hits are available.
    if (remove_seed_hits && tc.nFoundHits() <= backward_fit_min_hits) {
      remove_seed_hits = false;
    }

    // Stash HoTNodes at the end of m_hots.
    int stash_end = m_hots.size();
    int stash_pos = stash_end;

    int idx = tc.lastCcIndex();

    if (remove_seed_hits) {
      // Skip invalid hits that would now be at the head of the candidate.
      // Make sure to subtract / recount number of hits:
      // as this is rather involved, just call addHitIdx() repeatedly so counts
      // of holes get updated correctly.
      // Though one should not care super much ... it's only relevant for relative scores
      // and here we are trimming everything down to a single candidate.

      int n_hits_to_pick = std::max(tc.nFoundHits() - tc.getNSeedHits(), backward_fit_min_hits);
      while (n_hits_to_pick > 0) {
        m_hots[--stash_pos] = m_hots[idx];
        if (m_hots[idx].m_hot.index >= 0)
          --n_hits_to_pick;
        idx = m_hots[idx].m_prev_idx;
      }

      m_hots_size = 0;
      m_hots.clear();
      tc.setLastCcIndex(-1);
      tc.setNFoundHits(0);
      tc.setNMissingHits(0);
      tc.setNInsideMinusOneHits(0);
      tc.setNTailMinusOneHits(0);
      while (stash_pos != stash_end && m_hots[stash_pos].m_hot.index < 0)
        ++stash_pos;
      while (stash_pos != stash_end) {
        HoTNode &hn = m_hots[stash_pos];
        tc.addHitIdx(hn.m_hot.index, hn.m_hot.layer, hn.m_chi2);
        ++stash_pos;
      }
    } else {
      while (idx != -1) {
        m_hots[--stash_pos] = m_hots[idx];
        idx = m_hots[idx].m_prev_idx;
      }

      // If we are not removing seed_hits, track is good as it is,
      // just fixup m_hots and t.lastCcIndex.
      int pos = 0;
      while (stash_pos != stash_end) {
        m_hots[pos].m_hot = m_hots[stash_pos].m_hot;
        m_hots[pos].m_chi2 = m_hots[stash_pos].m_chi2;
        m_hots[pos].m_prev_idx = pos - 1;
        ++pos;
        ++stash_pos;
      }
      m_hots.resize(pos);
      m_hots_size = pos;
      tc.setLastCcIndex(pos - 1);
    }
  }

  void CombCandidate::beginBkwSearch() {
    // Assumes compactifyHitStorageForBestCand() has already been called.
    //
    // This is to be called before backward-search to start with a single
    // input candidate for backward combinatorial search.
    //
    // m_state and m_pickup_layer are also set.

    TrackCand &tc = m_trk_cands[0];

    m_state = Dormant;
    m_pickup_layer = m_hots[0].m_hot.layer;
    m_lastHitIdx_before_bkwsearch = tc.lastCcIndex();
    m_nInsideMinusOneHits_before_bkwsearch = tc.nInsideMinusOneHits();
    m_nTailMinusOneHits_before_bkwsearch = tc.nTailMinusOneHits();
    tc.setLastCcIndex(0);
    tc.setNInsideMinusOneHits(0);
    tc.setNTailMinusOneHits(0);
  }

  void CombCandidate::repackCandPostBkwSearch(int i) {
    // Called during filtering following backward search when a TrackCand's
    // front hits need to be reindexed.
    // mergeCandsAndBestShortOne() has already been called (from MkBuilder::FindXxx()).
    // NOTES:
    // 1. Should only be called once for each i (flag/bit to allow multiple calls can be added).
    // 2. Alternatively, CombCand could provide hit iterator/exporter that would handle this correctly.

    TrackCand &tc = m_trk_cands[i];

    int curr_idx = tc.lastCcIndex();
    if (curr_idx != 0) {
      int last_idx = -1, prev_idx;
      do {
        prev_idx = m_hots[curr_idx].m_prev_idx;

        m_hots[curr_idx].m_prev_idx = last_idx;

        last_idx = curr_idx;
        curr_idx = prev_idx;
      } while (prev_idx != -1);
    }

    tc.setLastCcIndex(m_lastHitIdx_before_bkwsearch);
    tc.setNInsideMinusOneHits(m_nInsideMinusOneHits_before_bkwsearch + tc.nInsideMinusOneHits());
    tc.setNTailMinusOneHits(m_nTailMinusOneHits_before_bkwsearch + tc.nTailMinusOneHits());
  }

}  // namespace mkfit
