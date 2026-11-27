#include "MkFinderV2p2.h"

#include "MkBins.h"

#include <algorithm>

#if defined(MKFIT_STANDALONE)
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

#ifdef MKFIT_TRACE
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntConversions.h"
#endif

//#define DEBUG
#include "Debug.h"

// Counters of the current TrLayerSearch, empty without MKFIT_TRACE.
#ifdef MKFIT_TRACE
#define V2P2_TR_LS_COUNT(ls_id, field) (++mp_event->tr_layersearch(ls_id).field)
#else
#define V2P2_TR_LS_COUNT(ls_id, field) ((void)0)
#endif

//==============================================================================
// How MkFinderV2p2 works, a reader's map.
//
// Words: a seed starts a CombCandidate, which holds up to maxCandsPerSeed
// TrackCands, the candidate tracks grown from that seed. The layer plan is the
// ordered list of layers searched, one per eta region; a seed is picked up at the
// plan entry of its last hit. The design notes, doc/MkFinderV2p2-DesignNotes.md,
// start with a list of the other words used here.
//
// MkBuilder::findTracksStandardv2p2() runs one finder over a block of seeds and
// the layer plan: begin_layer(), process_layer(), end_layer() per plan entry. A
// plan entry is one layer, or two sub-layers (m_layer, m_layer_sec) such as the
// two sensors of a PS / 2S module.
//
// The default plan is split: each sensor of a module pair is its own plan entry,
// searched from the state after the previous entry, so a missed sensor gets its
// own hole. A paired entry, one for both sensors with a shared window and one
// in-layer search over both, is supported but not used by default; its holes are
// per entry, not per sensor. The hole and overlap-count code, and an option off
// by default (InLayer::sister_hole_drop), are written for the split plan.
//
// The objects, per layer:
//   CCandRep        one CombCandidate (one seed) while it is finding, from
//                   pickup until all its TrackCands have stopped.
//   CandIncoming    one of its TrackCands entering this layer: the search
//                   window, the pre-selected hits, those hits in path order.
//                   It lives for one layer.
//   PassedHitMatch  one hit that passed pre-selection for a CandIncoming, with
//                   the candidate's state at the crossing with the hit's module
//                   plane (mixed_state). Its dalpha, the turning angle there,
//                   orders the hits along the path.
//   CandNode        one hit taken in this layer and the state after its Kalman
//                   update, with a pointer to its CandIncoming and the index of
//                   the node before it on the path (-1 for the first hit).
//                   Following the links back from any node gives one path; the
//                   nodes of one CandIncoming form a tree, its paths sharing
//                   their beginnings. All nodes sit in one vector, m_node_arena,
//                   linked by index, until their CombCandidate has been selected.
//   layer step      what one candidate does in one layer, scored as one additive
//                   term (V2p2Score.h).
//
// One CombCandidate in one layer, as if it were alone:
//   for each TrackCand not stopped, after the stopping cuts    begin_next_Ccrep_in_layer()
//     propagate to the layer's two bounding surfaces           prop_to_layer_edges()
//     window from the covariance, binnor bin ranges            determine_search_windows()
//     in / at the edge of / outside the sensitive region       determine_wsr()
//     each hit in the bins: line pre-cut, then cross its       select_hits(), preselect_hit_batch()
//       module plane and cut
//       keep the best max_presel_hits per sub-layer by dphi
//     order the kept hits along the path                       order_hits_along_path()
//     paths: the subsequences of those hits in path order,     expand_in_layer(), rk_*()
//       up to max_hits_per_layer hits, one per module, each
//       hit Kalman-updated and accepted on its chi2
//   compete on score: every path, one hole per incoming        select_and_materialise()
//     candidate, the TrackCands not taken in; the best
//     maxCandsPerSeed become the CombCandidate's TrackCands
//
// What the code adds to that, for speed:
//   Batching. Up to NN incoming candidates, from any CombCandidates, pass
//     propagation, window and pre-selection together: LayerBatch is NN
//     candidates wide, HitBatch NN (candidate, hit) pairs. The Kalman updates
//     of the paths go into two batches that fire when all NN lanes are filled,
//     or at the end of the layer, and stay open across layer batches
//     ("running Kalman", rk_): depth 0 takes the plane crossing
//     found in pre-selection, deeper steps solve for the plane from the parent
//     path's state. Accepted nodes go onto m_deep_queue and are turned into the
//     next depth's items there: a worklist, not recursion.
//   Completion. With batches mixing candidates, a CombCandidate's paths finish
//     at different times. Its CCandRep counts m_n_queued (incoming candidates
//     not yet through pre-selection) and m_n_inflight (Kalman items not yet run,
//     plus nodes not yet expanded; a child is counted before its parent is
//     released, so the count cannot pass through zero early). With both at zero
//     it can be selected right after the batch (InLayer::early_select, on by
//     default), and its nodes reused through m_node_free; otherwise it is
//     selected in end_layer().
//     m_ccrep_ready is only a hint; the counts are checked again before selecting.
//   Storage. m_node_arena is rewound at end_layer() and keeps its capacity, so
//     after a few layers it no longer allocates.
//
// The reasons behind each piece are in doc/MkFinderV2p2-DesignNotes.md.
//==============================================================================

namespace mkfit {

  using namespace Config::V2p2;

  void MkFinderV2p2::setup(const MkJob* job,
                           EventOfCombCandidates& eoccs,
                           int seed_begin,
                           int seed_end,
                           SteeringParams::iterator& sp_it,
                           const Event* ev) {
    mp_job = job;
    mp_steeringparams_iter = &sp_it;
    mp_event = ev;

    m_batch_mgr.setup(eoccs, seed_begin, seed_end);

    // Assure all ccands are dormant and have a single tcand.
    for (auto& ccand : m_batch_mgr) {
      if (ccand.state() != CombCandidate::Dormant)
        throw std::runtime_error(
            "CombCandidate is expected to be dormant "
            "on entry into combinatorial search.");
      if (ccand.size() != 1)
        throw std::runtime_error(
            "CombCandidate is expected to have a single "
            "TrackCand on entry into combinatorial search.");
    }
    m_batch_mgr.m_n_dormant = m_batch_mgr.n_total();
  }

  float MkFinderV2p2::win_dphi_trk_fac() const {
    const float f = mp_job->v2p2().window_dphi_trk_fac;
    return f > 0.0f ? f : Window::dphi_trk_fac;
  }

  float MkFinderV2p2::win_dq_trk_fac() const {
    const float f = mp_job->v2p2().window_dq_trk_fac;
    return f > 0.0f ? f : Window::dq_trk_fac;
  }

  void MkFinderV2p2::reset() {
    m_batch_mgr.release();
    mp_job = nullptr;
    mp_steeringparams_iter = nullptr;
    mp_event = nullptr;

    // Finders are reused across seed blocks and events, and the queue holds
    // pointers into CCandRep::m_cands. process_layer() asserts it is empty on exit.
    m_active_ccreps.clear();
    m_cand_queue.clear();
    m_cand_queue_head = 0;
  }

  int MkFinderV2p2::awaken_candidates() {
    // Awaken dormant ccands with hits on current layer for processing on the next layer.
    int count = 0;
    const LayerControl& lc = mp_steeringparams_iter->layer_control();
    for (auto& ccand : m_batch_mgr) {
      // Pick up at either sub-layer of a paired entry; the search starts at the
      // next entry. See doc/MkFinderV2p2-DesignNotes.md, "Candidate pickup and
      // stopping cuts".

      if (ccand.state() == CombCandidate::Dormant &&
          (ccand.pickupLayer() == lc.m_layer || ccand.pickupLayer() == lc.m_layer_sec)) {
        ccand.setState(CombCandidate::Finding);
        // Start the in-flight score accumulator. Any constant seed term is
        // common to every candidate of this seed and so cancels in the per-seed
        // selection; the cross-seed comparison is done at the end, on the
        // official score.
        for (int ic = 0; ic < (int)ccand.size(); ++ic)
          ccand[ic].setScore(0.0f);
        m_active_ccreps.emplace_back(ccand);
        dprintf("MkFinderV2p2::awaken_candidates N_TrackCands=%d\n", (int)ccand.size());
        ++count;
      }
    }
    m_batch_mgr.m_n_dormant -= count;
    m_batch_mgr.m_n_finding += count;

    dprintf("MkFinderV2p2::awaken_candidates woke up %d cands\n", count);

    return count;
  }

  void MkFinderV2p2::begin_layer() {
    dprintf("MkFinderV2p2::begin_layer Expecting %d active ccands\n", (int)m_active_ccreps.size());

#ifdef DEBUG
    {
      int i = 1;
      for (auto& ccrep : m_active_ccreps) {
        dprintf("  %2d. seed-idx=%d, n_tcands=%d\n", i, ccrep.m_ccand.seed_origin_index(), (int)ccrep.m_ccand.size());
        ++i;
      }
    }
#endif

    m_active_ccreps_pos = m_active_ccreps.begin();

    {  // Setup m_rz_limits.
      SteeringParams::iterator& spi = *mp_steeringparams_iter;

      const bool is_double_layer = spi->has_second_layer();
      const LayerInfo& LI_p = mp_job->m_trk_info[spi->m_layer];
      const LayerInfo& LI_s = mp_job->m_trk_info[is_double_layer ? spi->m_layer_sec : 0];

      if (is_double_layer) {
        assert(LI_p.is_barrel() == LI_s.is_barrel());
        m_rz_limits.setup(LI_p, LI_s, spi.is_outward());
      } else {
        m_rz_limits.setup(LI_p, spi.is_outward());
      }
    }
  }

  void MkFinderV2p2::process_layer() {
    // Pull CombCandidates into the layer until NN candidates are queued, then
    // drain the queue NN at a time. See doc/MkFinderV2p2-DesignNotes.md,
    // "Layer processing".
    while (any_Ccreps_to_begin()) {
      while (!enough_work_for_batch() && any_Ccreps_to_begin()) {
        begin_next_Ccrep_in_layer();
      }
      while (enough_work_for_batch() || (!any_Ccreps_to_begin() && any_work_for_batch())) {
        process_layer_batch();
      }
    }

    // Drained by construction: the inner loop's second clause runs until empty
    // once there is nothing left to pull in.
    assert(cand_queue_size() == 0 && "pre-select queue not drained by process_layer()");

    // The in-layer Kalman batches hold partial work across layer batches; empty
    // them before end_layer() selects whatever is left.
    rk_drain();
  }

  void MkFinderV2p2::end_layer() {
    int count = 0;
    auto ai = m_active_ccreps.begin();
    while (ai != m_active_ccreps.end()) {
      if (!ai->m_selected)
        select_and_materialise(*ai);

      ai->m_cands.clear();
      ai->m_nodes.clear();

      // A CombCandidate is finished once every TrackCand under it has stopped.
      bool is_finished = false;
      if (Policy::use_stop_cuts || Policy::use_hole_limits) {
        CombCandidate& cc = ai->m_ccand;
        is_finished = true;
        for (int ic = 0; ic < (int)cc.size(); ++ic) {
          if (cc[ic].getLastHitIdx() != Hit::kHitStopIdx) {
            is_finished = false;
            break;
          }
        }
        if (is_finished) {
          cc.setState(CombCandidate::Finished);
          V2P2_COUNT(n_ccand_retired);
        }
      }
      if (is_finished) {
        auto bi = ai++;
        m_active_ccreps.erase(bi);
        ++count;
        continue;
      }
      ++ai;
    }
    m_batch_mgr.m_n_finding -= count;

    // One rewind for the whole layer, keeping capacity: after a few layers the
    // arena is at high water and stops allocating.
    m_node_arena.clear();
    m_node_free.clear();

    m_rz_limits.reset();

    dprintf("MkFinderV2p2::end_layer %d cands finished\n", count);

    if (m_batch_mgr.has_dormant_ccands())
      awaken_candidates();
  }

  //----------------------------------------------------------------------------
  // Layer control: pull CombCandidates into the layer, queue their TrackCands,
  // and run the queue through process_layer_batch(); the candidate-stopping cuts
  // and the fake HoT of a candidate that takes no hit.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::begin_next_Ccrep_in_layer() {
    CCandRep& ccrep = *m_active_ccreps_pos;
    CombCandidate& ccand = ccrep.m_ccand;

    // Beam width for this layer, per seed: see InLayer::late_max_cands. The
    // TrackCands are in score order from the previous selection, so dropping
    // the tail here is the selection having kept fewer.
    // The capacity is reserved for the larger of the forward and backward widths
    // (MkJob::max_max_cands()); the search direction's own maxCandsPerSeed applies.
    ccrep.m_max_cands = std::min((int)ccand.capacity(), mp_job->params_cur().maxCandsPerSeed);
    if (InLayer::late_max_cands > 0 && !ccand.empty() && ccand[0].nFoundHits() > InLayer::late_max_cands_hits &&
        !long_step_to_layer(ccand[0])) {
      ccrep.m_max_cands = std::min(ccrep.m_max_cands, (int)InLayer::late_max_cands);
      V2P2_COUNT(n_late_max_cands);
      if ((int)ccand.size() > ccrep.m_max_cands) {
        V2P2_COUNT_ADD(n_late_max_cands_dropped, (int)ccand.size() - ccrep.m_max_cands);
        ccand.resize(ccrep.m_max_cands);
      }
    }

    for (int ic = 0; ic < (int)ccand.size(); ++ic) {
      TrackCand& tcand = ccand[ic];

      if (tcand.getLastHitIdx() == Hit::kHitStopIdx)
        continue;

      // Stopping cuts, at pull-in: see stop_cuts_at_pickup().
      if (Policy::use_stop_cuts) {
        const TrackCand::StopReason_e sr = stop_cuts_at_pickup(tcand);
        if (sr != TrackCand::SR_NotStopped) {
          tcand.setStopReason(sr);
          tcand.addHitIdx(Hit::kHitStopIdx, m_rz_limits.layer_info_1().layer_id(), 0.0f);
          if (sr == TrackCand::SR_MinPt)
            V2P2_COUNT(n_stop_minpt);
          else
            V2P2_COUNT(n_stop_looper);
          continue;
        }
      }

      // "Layer already passed in z" pre-check, before any propagation. A failed
      // check skips the layer without recording anything, as WSR_Outside does.

      if (!m_rz_limits.rz_quadrant_check(tcand.z(), tcand.pz())) {
        V2P2_COUNT(n_quadrant_skip);
        continue;
      }

      // Create and register a CandIncoming for processing. m_cands does not
      // reallocate, see the CCandRep constructor.
      {
        CandIncoming& cand = ccrep.m_cands.emplace_back(&ccrep, ic);
        m_cand_queue.push_back(&cand);
      }
    }
    ccrep.m_n_queued = (int)ccrep.m_cands.size();
    ccrep.m_selected = false;
    ++m_active_ccreps_pos;
  }

  //----------------------------------------------------------------------------
  // process_layer_batch() -- one batch of up to NN candidates through the layer.
  // LayerBatch is NN candidates wide, HitBatch is NN (candidate, hit) pairs wide.
  // See doc/MkFinderV2p2-DesignNotes.md, "Layer processing".
  //----------------------------------------------------------------------------

  void MkFinderV2p2::process_layer_batch() {
    LayerBatch b;

    prop_to_layer_edges(b);       // pop the queue, propagate to the layer edges
    determine_search_windows(b);  // covariance -> dphi/dq windows -> bin ranges
    select_hits(b);               // walk the bins, pre-select, reduce in a pqueue
    order_hits_along_path(b);     // pqueue -> m_layer_hits, step-ordered
    // The paths stay in the arena until their CombCandidate is selected in
    // select_and_materialise(): right after the batch with InLayer::early_select,
    // else in end_layer().
    expand_in_layer(b);  // grow the paths, as deep as the full Kalman batches allow
    if (InLayer::early_select)
      select_completed_ccreps(b);
  }

  //------------------------------------------------------------------------------
  // stop_cuts_at_pickup() -- minPtCut and the looper stop, which need only the
  // candidate's own state. The looper test covers both wrap images of
  // |posPhi - momPhi|. See doc/MkFinderV2p2-DesignNotes.md,
  // "Candidate pickup and stopping cuts".
  //----------------------------------------------------------------------------

  TrackCand::StopReason_e MkFinderV2p2::stop_cuts_at_pickup(const TrackCand& tc) const {
    if (tc.pT() < mp_job->params_cur().minPtCut)
      return TrackCand::SR_MinPt;

    // Forward search only: going inward the track is leaving the turning region,
    // so a large angle there is not evidence that it is about to curl up.
    if (mp_steeringparams_iter->is_outward() && tc.pT() < Policy::looper_max_pt &&
        tc.posRsq() > Policy::looper_min_r * Policy::looper_min_r) {
      const float max_ang = Policy::looper_max_angle;
      const float dphi = std::abs(tc.posPhi() - tc.momPhi());
      if (dphi > max_ang && dphi < Const::TwoPI - max_ang)
        return TrackCand::SR_Looper;
    }

    return TrackCand::SR_NotStopped;
  }

  // Straight-line path length from the candidate's state (its last hit) to the
  // layer's near surface, against InLayer::late_max_cands_step_cm. Barrel: radial gap
  // times p / pT; endcap: z gap times p / |pz|. Curvature is ignored, so the
  // barrel path is underestimated for strongly bending tracks.
  bool MkFinderV2p2::long_step_to_layer(const TrackCand& tc) const {
    if (InLayer::late_max_cands_step_cm <= 0.0f)
      return false;
    const LayerInfo& li = m_rz_limits.layer_info_1();
    float s;
    if (li.is_barrel()) {
      const float r = tc.posR();
      const float dr = std::max({li.rin() - r, r - li.rout(), 0.0f});
      s = dr * tc.p() / std::max(1e-3f, tc.pT());
    } else {
      const float z = tc.z();
      const float dz = std::max({li.zmin() - z, z - li.zmax(), 0.0f});
      s = dz * tc.p() / std::max(1e-3f, std::abs(tc.pz()));
    }
    if (s > InLayer::late_max_cands_step_cm) {
      V2P2_COUNT(n_late_max_cands_long);
      return true;
    }
    return false;
  }

  //----------------------------------------------------------------------------
  // fake_hit_index() -- the fake HoT for a candidate that took no hit: the hole
  // limits choose miss or stop, then the WSR overrides with edge or gap, in
  // V1's order.
  //----------------------------------------------------------------------------

  int MkFinderV2p2::fake_hit_index(const TrackCand& tc, const WSR_Result& wsr) const {
    const auto& iter_params = mp_job->params_cur();

    int fake_hit_idx = Hit::kHitMissIdx;

    if (Policy::use_hole_limits &&
        (tc.nAllMinusOneHits() >= iter_params.maxHolesPerCand || tc.nTailMinusOneHits() >= iter_params.maxConsecHoles))
      fake_hit_idx = Hit::kHitStopIdx;

    if (Policy::use_wsr) {
      if (wsr.m_wsr == WSR_Edge)
        fake_hit_idx = Hit::kHitEdgeIdx;
      else if (wsr.m_in_gap)
        fake_hit_idx = Hit::kHitInGapIdx;
    }

    return fake_hit_idx;
  }

  //----------------------------------------------------------------------------
  // Phase 1 -- which candidates, and where do they meet the layer.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::prop_to_layer_edges(LayerBatch& b) {
    MkBins& B = b.B;
    CandIncoming** cand_ptrs = b.cand;

    const int N_proc = b.N_proc = std::min(NN, cand_queue_size());
    // MkBins computes at full width, see pad_free_lanes().
    B.m_n_proc = NN;

    dprintf("MkFinderV2p2::process_layer_batch work queue is %d, would process %d of them (NN=%d)\n",
            cand_queue_size(),
            N_proc,
            NN);

    MPlexQF phi(0.0f);
    MPlexQI chg(0);
    MPlexQF bf(Config::Bfield);
    const bool param_b = Window::mini_prop_param_b == 1 ||
                         (Window::mini_prop_param_b == 2 &&
                          mp_job->m_trk_info.prop_config().finding_inter_layer_pflags.use_param_b_field);

    for (int i = 0; i < N_proc; ++i) {
      CandIncoming& cand = *m_cand_queue[m_cand_queue_head++];
      cand_ptrs[i] = &cand;
      TrackCand& tc = cand.tcand();
      --cand.mp_ccrep->m_n_queued;

      // Copy in x, y,z, invpT, theta.
      B.m_isp.copyIn_partial_track_state(i, tc.state());
      phi[i] = tc.momPhi();
      chg[i] = tc.charge();
      if (param_b)
        bf[i] = Config::bFieldFromZR(tc.z(), std::hypot(tc.x(), tc.y()));
    }
    for (int i = N_proc; i < NN; ++i)
      B.m_isp.copyIn_partial_track_state(i, cand_ptrs[0]->tcand().state());
    pad_free_lanes(N_proc, phi, chg, bf);
    B.m_isp.init_momentum_vec_and_k(phi, chg, bf);
    if (m_cand_queue_head == (int)m_cand_queue.size()) {
      m_cand_queue.clear();
      m_cand_queue_head = 0;
    }

    // m_sp1 at the layer's near bounding surface, m_sp2 at the far one.
    B.prop_to_limits_in_order(m_rz_limits);
  }

  //----------------------------------------------------------------------------
  // Phase 2 -- the search windows. The position block of the track covariance is
  // transported to m_sp2 (MkBins::transport_position_cov()); from it come
  // dphi_track and dq_track, and from those the WSR verdict and the binnor
  // ranges. The function also builds the Hermite cubic through the two layer
  // crossings. See doc/MkFinderV2p2-DesignNotes.md, "Search window".
  //----------------------------------------------------------------------------

  void MkFinderV2p2::determine_search_windows(LayerBatch& b) {
    SteeringParams::iterator& spi = *mp_steeringparams_iter;
    const int N_proc = b.N_proc;
    CandIncoming** cand_ptrs = b.cand;
    MkBins& B = b.B;
    MkBinTrackCovExtract& TCE = b.TCE;
    MkBinLimits& BL_p = b.BL_p;
    MkBinLimits& BL_s = b.BL_s;
    mini_propagators::Hermite3D& H = b.H;
#ifdef MKFIT_TRACE
    int* tr_layersearch_ids = b.tr_layersearch_ids;
#endif

    // Covariance position block at m_sp2, transported from the previous hit.
    MPlexLV par0;
    MPlexLS err0;
    for (int i = 0; i < N_proc; ++i) {
      const TrackCand& tc = cand_ptrs[i]->tcand();
      par0.copyIn(i, tc.parArray());
      err0.copyIn(i, tc.errArray());
    }
    for (int i = N_proc; i < NN; ++i) {
      par0.copyIn(i, par0, 0);
      err0.copyIn(i, err0, 0);
    }
    B.transport_position_cov(par0, err0, TCE);

    B.determine_bin_windows(TCE);

    // The WSR edge fuzz uses dq_track, so the verdict follows determine_bin_windows().
    determine_wsr(b);

    B.find_bin_ranges(mp_job->m_event_of_hits[spi->m_layer], BL_p, win_dphi_trk_fac(), win_dq_trk_fac());

    if (m_rz_limits.m_is_double) {
      B.find_bin_ranges(mp_job->m_event_of_hits[spi->m_layer_sec], BL_s, win_dphi_trk_fac(), win_dq_trk_fac());
    }

#ifdef DEBUG
    // clang-format off
    for (int i = 0; i < N_proc; ++i) {
      dprintf("%d: BinCheck Prim %c %+8.6f %+8.6f | %3d %3d || %+8.6f %+8.6f | %2d %2d\n",
              i, m_rz_limits.m_is_barrel ? 'B' : 'E',
              B.m_phi_center[i], B.m_phi_delta[i], BL_p.p1[i], BL_p.p2[i],
              B.m_q_min[i], B.m_q_max[i], BL_p.q1[i], BL_p.q2[i]);
      if (m_rz_limits.m_is_double) {
        dprintf("%d: BinCheck Sec  %c %+8.6f %+8.6f | %3d %3d || %+8.6f %+8.6f | %2d %2d\n",
                i, m_rz_limits.m_is_barrel ? 'B' : 'E',
                B.m_phi_center[i], B.m_phi_delta[i], BL_s.p1[i], BL_s.p2[i],
                B.m_q_min[i], B.m_q_max[i], BL_s.q1[i], BL_s.q2[i]);
      }
    }
    // clang-format on
#endif

    // Hermite cubic through the two layer crossings, the trajectory model for the
    // per-hit plane solve. See the design notes, "Layer crossings and the Hermite
    // cubic".
    namespace mp = mini_propagators;

    H.calculate_coeffs(B.m_sp1, B.m_sp2, B.m_isp.inv_k);

#ifdef MKFIT_TRACE
    for (int i = 0; i < N_proc; ++i) {
      const CandIncoming& cand = *cand_ptrs[i];
      TrLayerSearch ls;
      ls.state_id = cand.tcand().m_trace_state_id;
      ls.wsr = (signed char)cand.m_wsr.m_wsr;
      ls.wsr_in_gap = cand.m_wsr.m_in_gap;
      fill_layer(ls, *spi, m_rz_limits);
      fill_window(ls, B, TCE, i);
      fill_bins(ls, BL_p, m_rz_limits.m_is_double ? &BL_s : nullptr, i);
      tr_layersearch_ids[i] = mp_event->trace_layersearch(std::move(ls)).id;
    }
#endif

#ifdef MKFIT_TRACE_PROP_COMPARE
    // Hermite vs helix in mid-layer, where the cubic is furthest from the two
    // endpoints it interpolates (at t = 0 and 1 it reproduces them by
    // construction, so those are roundoff checks only and not worth taking).
    // Evaluate the cubic at t = 0.5, then propagate exactly to the bounding
    // surface -- r in the barrel, z in the endcap -- that the cubic's midpoint
    // landed on, so both points sit on the same surface and the difference is a
    // pure in-surface deviation.
    {
      MPlexQF hx, hy, hz;
      H.evaluate(0.5f, hx, hy, hz);
      mp::StatePlex sp_mid;
      if (m_rz_limits.m_is_barrel) {
        MPlexQF r_mid = Matriplex::hypot(hx, hy);
        B.m_isp.propagate_to_r(mp::PA_Exact, r_mid, sp_mid, false, N_proc);
      } else {
        B.m_isp.propagate_to_z(mp::PA_Exact, hz, sp_mid, false, N_proc);
      }
      for (int i = 0; i < N_proc; ++i) {
        EVec3 dev(hx[i] - sp_mid.x[i], hy[i] - sp_mid.y[i], hz[i] - sp_mid.z[i]);
        // clang-format off
        dprintf("%d: HermiteMid dev %+9.6f %+9.6f %+9.6f  (|d| = %.6f, dalpha %.4f -> %.4f, derfac %.5f)\n",
                i, dev[0], dev[1], dev[2],
                std::sqrt(dev.Mag2()), B.m_sp1.dalpha[i], B.m_sp2.dalpha[i], H.m_Hderfac[i]);
        // clang-format on
#ifdef MKFIT_TRACE
        mp_event->tr_layersearch(tr_layersearch_ids[i]).hermite_mid_dev = dev;
#endif
      }
    }
#endif
  }

  //----------------------------------------------------------------------------
  // Phase 2b -- within-sensitive-region verdict per candidate: does the segment
  // between the two layer crossings lie in sensitive material. The fail flags
  // of the two crossings answer the radial half; the segment's q extent,
  // widened by Policy::wsr_n_sigma, answers the rest. See
  // doc/MkFinderV2p2-DesignNotes.md, "Within-sensitive-region verdict".
  //----------------------------------------------------------------------------

  void MkFinderV2p2::determine_wsr(LayerBatch& b) {
    // m_dq_track is 3 sigma.
    const float dq_to_n_sigma = Policy::wsr_n_sigma / 3.0f;

    const int N_proc = b.N_proc;
    MkBins& B = b.B;
    CandIncoming** cand_ptrs = b.cand;

    const bool is_barrel = m_rz_limits.m_is_barrel;
    const float q_lo = is_barrel ? m_rz_limits.m_zmin : m_rz_limits.m_rin;
    const float q_hi = is_barrel ? m_rz_limits.m_zmax : m_rz_limits.m_rout;
    const LayerInfo& LI = m_rz_limits.layer_info_1();

    for (int i = 0; i < N_proc; ++i) {
      const bool f_entry = B.m_sp1.fail_flag[i] != 0;
      const bool f_exit = B.m_sp2.fail_flag[i] != 0;

      WSR_Result w;

      if (f_entry && f_exit) {
        // Clear miss: the track does not reach the layer at all.
        w = WSR_Result(WSR_Outside, false);
      } else {
        const float dq = dq_to_n_sigma * B.m_dq_track[i];
        const float qa = B.m_q_min[i], qb = B.m_q_max[i];

        if (qb < q_lo - dq || qa > q_hi + dq)
          w = WSR_Result(WSR_Outside, false);
        else if (qa > q_lo + dq && qb < q_hi - dq)
          w = WSR_Result(WSR_Inside, false);
        else
          w = WSR_Result(WSR_Edge, false);

        // Near miss -- in the layer, but turning round inside it.
        if ((f_entry || f_exit) && w.m_wsr == WSR_Inside)
          w.m_wsr = WSR_Edge;

        // Endcap discs have a hole at small r. Sitting wholly inside it is a
        // miss with a reason; touching it is an edge, and either way m_in_gap
        // records that the absence of a hit is explained.
        if (!is_barrel && LI.has_r_range_hole() && w.m_wsr != WSR_Outside) {
          const float h_lo = LI.hole_r_min(), h_hi = LI.hole_r_max();
          if (qa > h_lo + dq && qb < h_hi - dq) {
            w = WSR_Result(WSR_Outside, true);
          } else if (qb > h_lo - dq && qa < h_hi + dq) {
            w.m_wsr = WSR_Edge;
            w.m_in_gap = true;
          }
        }
      }

      cand_ptrs[i]->m_wsr = w;

      // clang-format off
      dprintf("%d: WSR %d in_gap %d  (q %.3f - %.3f vs layer %.3f - %.3f, dq3 %.4f, fail %d/%d)\n",
              i, w.m_wsr, (int)w.m_in_gap, B.m_q_min[i], B.m_q_max[i], q_lo, q_hi,
              B.m_dq_track[i], B.m_sp1.fail_flag[i], B.m_sp2.fail_flag[i]);
      // clang-format on
    }
  }

  //----------------------------------------------------------------------------
  // Phase 3 -- walk the bin ranges and pre-select hits.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::select_hits(LayerBatch& b) {
    SteeringParams::iterator& spi = *mp_steeringparams_iter;
    const int N_proc = b.N_proc;
    MkBins& B = b.B;
    MkBinLimits& BL_p = b.BL_p;
#ifdef MKFIT_TRACE
    int* tr_layersearch_ids = b.tr_layersearch_ids;
#endif

    HitBatch hb;
    int& fill_pos = hb.fill_pos;
    auto& is_plex = hb.is_plex;
    auto& cand_idcs = hb.cand_idcs;
    auto& hit_idcs = hb.hit_idcs;
    auto& hit_orig_idcs = hb.hit_orig_idcs;

    // The iteration's window factors, read once per call.
    const float dphi_trk_fac = win_dphi_trk_fac(), dq_trk_fac = win_dq_trk_fac();

    // Both sub-layers, each into its own reduction queue. See the design notes,
    // "Reduction and hit ordering".
    const int n_sub = m_rz_limits.m_is_double ? 2 : 1;
    for (int sub = 0; sub < n_sub; ++sub) {
      const bool is_sec_layer = (sub == 1);
      const int lay = is_sec_layer ? spi->m_layer_sec : spi->m_layer;
      const auto& L = mp_job->m_event_of_hits[lay];
      const auto& iteration_hit_mask = mp_job->get_mask_for_layer(lay);
      const auto& BL = is_sec_layer ? b.BL_s : BL_p;
      fill_pos = 0;

      for (int i = 0; i < N_proc; ++i) {
        using bidx_t = LayerOfHits::bin_index_t;
        using bcnt_t = LayerOfHits::bin_content_t;

        // The track does not reach this layer.
        if (Policy::use_wsr && b.cand[i]->m_wsr.m_wsr == WSR_Outside)
          continue;

        // Line pre-cut, per candidate: the straight line m_sp1 -> m_sp2 in
        // (qbar, q) and (qbar, phi), and the hit-independent tolerance terms.
        // See doc/MkFinderV2p2-DesignNotes.md, "Line pre-cut".
        const bool barrel = m_rz_limits.m_is_barrel;
        bool pc_on = PreCut::q || PreCut::phi;
        float pc_qb1 = 0, pc_q1 = 0, pc_gq = 0, pc_phi1 = 0, pc_gphi = 0, pc_tol_q = 0, pc_tol_phi = 0;
        if (pc_on) {
          const float x1 = B.m_sp1.x[i], y1 = B.m_sp1.y[i], z1 = B.m_sp1.z[i];
          const float x2 = B.m_sp2.x[i], y2 = B.m_sp2.y[i], z2 = B.m_sp2.z[i];
          const float r1 = std::hypot(x1, y1), r2 = std::hypot(x2, y2);
          pc_qb1 = barrel ? r1 : z1;
          pc_q1 = barrel ? z1 : r1;
          const float dqb = (barrel ? r2 : z2) - pc_qb1;
          if (std::abs(dqb) < 1e-4f) {
            pc_on = false;  // no span to interpolate along
          } else {
            pc_gq = std::clamp(((barrel ? z2 : r2) - pc_q1) / dqb, -20.0f, 20.0f);
            pc_phi1 = B.m_phi1[i];
            pc_gphi = std::clamp(squashPhiGeneral(B.m_phi2[i] - pc_phi1) / dqb, -20.0f, 20.0f);
            pc_tol_q = PreCut::dq_slack * dq_trk_fac * B.m_dq_track[i] * (1.0f + pc_gq * pc_gq);
            pc_tol_phi = PreCut::dphi_slack * Window::capped_dphi(dphi_trk_fac * B.m_dphi_track[i]);
          }
        }

        // The region rho counts over and divides by, see the end of this function: the
        // window with the cut's padding of this sub-layer, as the fetch.
        const float phi_hit_term =
            Window::phi_per_hit ? Window::dphi_hit_fac * L.max_hit_phi_half_extent() : Window::dphi_flat_rad;
        const float rho_dphi = Window::capped_dphi(dphi_trk_fac * B.m_dphi_track[i] + phi_hit_term);
        const float rho_dq = dq_trk_fac * B.m_dq_track[i] + Window::dq_hit_fac * L.max_hit_q_half_length();
        b.rho_dphi[i] = std::max(b.rho_dphi[i], rho_dphi);
        b.rho_dq[i] = std::max(b.rho_dq[i], rho_dq);
        // Centre and half-width from MkBins: when the two crossings lie on both sides of
        // phi = +-pi, m_phi_min and m_phi_max hold them swapped for the cyclic bin walk and
        // are not the ends of an interval.
        const float rho_phi_c = B.m_phi_center[i];
        const float rho_phi_half = B.m_phi_delta[i] + rho_dphi;
        const float rho_q_lo = B.m_q_min[i] - rho_dq;
        const float rho_q_hi = B.m_q_max[i] + rho_dq;

        for (bidx_t qi = BL.q1[i]; qi != BL.q2[i]; ++qi) {
          for (bidx_t pi = BL.p1[i]; pi != BL.p2[i]; pi = L.phiMaskApply(pi + 1)) {
            auto pbi = L.phiQBinContent(pi, qi);
            for (bcnt_t hi = pbi.begin(); hi < pbi.end(); ++hi) {
              const unsigned int hi_orig = L.getOriginalHitIndex(hi);

              // clang-format off
              dprintf(" %d: P_HIT %3u %4u %5u [%5u]  %6.3f %6.3f %6.3f\n",
                      i, pi, qi, hi, hi_orig, L.hit_phi(hi), L.hit_q(hi), L.hit_qbar(hi));
              // clang-format on

              if (std::abs(squashPhiGeneral(L.hit_phi(hi) - rho_phi_c)) < rho_phi_half && L.hit_q(hi) > rho_q_lo &&
                  L.hit_q(hi) < rho_q_hi)
                ++b.n_scanned[i];
              V2P2_TR_LS_COUNT(tr_layersearch_ids[i], n_hits_scanned);

              if (iteration_hit_mask && (*iteration_hit_mask)[hi_orig]) {
                V2P2_TR_LS_COUNT(tr_layersearch_ids[i], n_hits_masked);
                continue;
              }

              if (pc_on) {
                const float dqb_h = L.hit_qbar(hi) - pc_qb1;
                const float qbar_term = barrel ? PreCut::qbar_fac * L.hit_qbar_half_extent(hi) : 0.0f;
                bool reject = false;
                if (PreCut::q) {
                  const float dev = std::abs(L.hit_q(hi) - (pc_q1 + dqb_h * pc_gq));
                  reject = dev > pc_tol_q + Window::dq_hit_fac * L.hit_q_half_length(hi) + std::abs(pc_gq) * qbar_term;
                }
                if (!reject && PreCut::phi) {
                  const float dev = std::abs(squashPhiGeneral(L.hit_phi(hi) - (pc_phi1 + dqb_h * pc_gphi)));
                  const float hit_term =
                      Window::phi_per_hit ? Window::dphi_hit_fac * L.hit_phi_half_extent(hi) : Window::dphi_flat_rad;
                  reject = dev > pc_tol_phi + hit_term + std::abs(pc_gphi) * qbar_term;
                }
                if (reject) {
                  V2P2_TR_LS_COUNT(tr_layersearch_ids[i], n_hits_precut);
                  continue;
                }
              }

              // The Hit is read in preselect_hit_batch().
              __builtin_prefetch(&L.refHit(hi_orig));

              cand_idcs[fill_pos] = i;
              hit_idcs[fill_pos] = hi;
              hit_orig_idcs[fill_pos] = hi_orig;
              is_plex.copyIn(fill_pos, B.m_isp, i);

              if (++fill_pos == NN) {
                preselect_hit_batch(b, hb, L, NN, is_sec_layer);
                fill_pos = 0;
              }
            }
          }
        }
      }
      if (fill_pos > 0) {
        preselect_hit_batch(b, hb, L, fill_pos, is_sec_layer);
      }
    }

    // Local hit density, hits per cm^2, over the cut region: the window widened by
    // the cut's track and largest-hit-extent terms as the fetch computes them, but
    // without bin rounding or spare bins. Hits are counted over the same region, not
    // over the pre-selected ones, so the estimate does not depend on the cut it is
    // about to feed. The phi extent becomes a length at the crossing radius. The count
    // can be 0 with a hit about to be scored, so the scored hit is counted too.
    for (int i = 0; i < N_proc; ++i) {
      const float r = std::max(0.1f, std::hypot(B.m_sp2.x[i], B.m_sp2.y[i]));
      // m_phi_delta is the half-width of the crossing span, also across phi = +-pi. The
      // region is at most the full turn.
      const float w_phi = std::min(2.0f * (B.m_phi_delta[i] + b.rho_dphi[i]), Const::TwoPI);
      const float w_q = (B.m_q_max[i] - B.m_q_min[i]) + 2.0f * b.rho_dq[i];
      const float area = std::max(1e-4f, w_phi * r * w_q);
      b.cand[i]->m_log_rho = std::log(std::max(1e-6f, (b.n_scanned[i] + 1.0f) / area));
#if defined(MKFIT_STANDALONE)
      // Diag::final_beam_purity: the hits in the density region are the expected
      // background of this layer step, per (seed, layer), forward search only.
      if (Diag::final_beam_purity && spi.is_outward())
        v2p2_final_beam_bg_record(
            b.cand[i]->ccand().seed_origin_index(), spi->m_layer, spi->m_layer_sec, b.n_scanned[i]);
#endif
    }
  }

  //----------------------------------------------------------------------------
  // surface_referenced_dq() -- the track's q error referenced to the hit's
  // module plane rather than to the fixed path length it was transported to:
  //     v = e_q - ((e_q.p^)/(n^.p^)) n^ ,   sigma_q^2 = v^T C v
  // with n^ the module normal. Returns 3 sigma_q, the convention of
  // MkBins::m_dq_track, which is also the fallback. See
  // doc/MkFinderV2p2-DesignNotes.md, "Search window".
  //----------------------------------------------------------------------------

  float MkFinderV2p2::surface_referenced_dq(float dq_track_fallback,
                                            const MkBinTrackCovExtract& TCE,
                                            int pi,
                                            const mini_propagators::StatePlex& h3_state,
                                            int h,
                                            const MPlex3V& module_norm,
                                            bool is_barrel) {
    const float px = h3_state.px[h], py = h3_state.py[h], pz = h3_state.pz[h];
    const float nx = module_norm(h, 0, 0), ny = module_norm(h, 1, 0), nz = module_norm(h, 2, 0);
    const float np = nx * px + ny * py + nz * pz;  // (n^.p) -- |p| cancels below
    // q direction: z in the barrel -- so e_q.p is just pz, with no dot product
    // and no hipo -- and r^ in the endcap.
    float ex = 0.0f, ey = 0.0f, ez = 1.0f, eqp = pz;
    bool ok = (np != 0.0f);
    if (ok && !is_barrel) {
      const float rr = hipo(h3_state.x[h], h3_state.y[h]);
      ok = (rr > 0.0f);
      if (ok) {
        ex = h3_state.x[h] / rr;
        ey = h3_state.y[h] / rr;
        ez = 0.0f;
        eqp = ex * px + ey * py;
      }
    }
    if (!ok)
      return dq_track_fallback;

    // f = (e_q.p^)/(n^.p^); |p| cancels, so no normalisation is needed. Clamped:
    // it diverges only at grazing incidence on the module, which is not an
    // operating point (the cluster is then too wide in phi to be a hit).
    const float f = std::clamp(eqp / np, -20.0f, 20.0f);
    const float v0 = ex - f * nx, v1 = ey - f * ny, v2 = ez - f * nz;
    const float var = v0 * v0 * TCE.m_cov_0_0[pi] + v1 * v1 * TCE.m_cov_1_1[pi] + v2 * v2 * TCE.m_cov_2_2[pi] +
                      2.0f * (v0 * v1 * TCE.m_cov_0_1[pi] + v0 * v2 * TCE.m_cov_0_2[pi] + v1 * v2 * TCE.m_cov_1_2[pi]);
    if (var <= 0.0f)
      return dq_track_fallback;
    return 3.0f * std::sqrt(var);
  }

  //----------------------------------------------------------------------------
  // Phase 3b -- pre-select one NN-wide batch of (candidate, hit) pairs.
  //
  // Solves the Hermite cubic onto each hit's own module plane, applies the
  // dq / dphi pre-selection cut, and pushes survivors into the candidate's
  // bounded priority queue.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::preselect_hit_batch(
      LayerBatch& b, HitBatch& hb, const LayerOfHits& L, int N_proc_hits, bool is_sec_layer) {
    MkBins& B = b.B;
    MkBinTrackCovExtract& TCE = b.TCE;
    CandIncoming** cand_ptrs = b.cand;
    mini_propagators::Hermite3D& H = b.H;
    namespace mp = mini_propagators;
    auto& is_plex = hb.is_plex;
    auto& cand_idcs = hb.cand_idcs;
    auto& hit_idcs = hb.hit_idcs;
    auto& hit_orig_idcs = hb.hit_orig_idcs;
#ifdef MKFIT_TRACE_PROP_COMPARE
    auto& h_plex = hb.h_plex;
#endif
#ifdef MKFIT_TRACE
    int* tr_layersearch_ids = b.tr_layersearch_ids;
#endif

    // The lanes from N_proc_hits up get copies of lane 0 below, so that the Newton
    // step in solve() and the incidence cut compute on valid numbers there; their
    // results are not read. See pad_free_lanes().
    MPlex3V module_pos;
    MPlex3V module_norm;
    mp::Hermite3D h3d;

#ifdef MKFIT_TRACE
    int tr_hitmatch_ids[NN];
#endif

    // Extract hit / target module data
    for (int h = 0; h < N_proc_hits; ++h) {
      const Hit& hit = L.refHit(hit_orig_idcs[h]);
      unsigned int mid = hit.detIDinLayer();
      const ModuleInfo& mi = L.layer_info().module_info(mid);
      module_pos.copyIn(h, mi.pos.Array());
      module_norm.copyIn(h, mi.zdir.Array());

      h3d.copyIn(h, H, cand_idcs[h]);

#ifdef MKFIT_TRACE
      CandIncoming& cand = *cand_ptrs[cand_idcs[h]];
      TrHitMatch hm = make_hitmatch(mp_event, cand.ccand().seed_origin_index(), L, hit_orig_idcs[h], is_sec_layer);
      hm.state_id = cand.tcand().m_trace_state_id;
      hm.search_id = tr_layersearch_ids[cand_idcs[h]];
      tr_hitmatch_ids[h] = mp_event->trace_hitmatch(std::move(hm)).id;
#endif
    }
    pad_free_lanes(N_proc_hits, module_pos, module_norm);
    for (int h = N_proc_hits; h < NN; ++h)
      h3d.copyIn(h, h3d, 0);

#ifdef MKFIT_TRACE_PROP_COMPARE
    is_plex.propagate_to_plane(mp::PA_Line, module_pos, module_norm, h_plex, true);
#endif

    mp::Hermite3DOnPlane h3dop;
    h3dop.init_coeffs(h3d, module_pos, module_norm);
    // NOTE: solve() takes two Newton steps, not an iteration to convergence.
    // d3 below is what says whether they were enough, so it goes into the trace
    // and not just into DEBUG.
    h3dop.solve();
#ifdef MKFIT_TRACE
    MPlexQF d3;
    h3dop.evaluate(h3dop.m_T, d3);  // residual distance to plane after the solve
#endif

    mp::StatePlex h3_state;
    h3d.evaluate(h3dop.m_T, h3_state);
    // h3_state.dalpha is interpolated per hit below.

    // cos^2 of the incidence on each hit's module, for the grazing cut. See
    // doc/MkFinderV2p2-DesignNotes.md, "Grazing incidence".
    MPlexQF cos2_inc(1.0f);
    if (Window::min_cos_incidence > 0.0f) {
#pragma omp simd
      for (int h = 0; h < NN; ++h) {
        const float pn = h3_state.px[h] * module_norm(h, 0, 0) + h3_state.py[h] * module_norm(h, 1, 0) +
                         h3_state.pz[h] * module_norm(h, 2, 0);
        const float p2 =
            h3_state.px[h] * h3_state.px[h] + h3_state.py[h] * h3_state.py[h] + h3_state.pz[h] * h3_state.pz[h];
        cos2_inc[h] = pn * pn / p2;
      }
    }
    const float min_cos2_inc = Window::min_cos_incidence * Window::min_cos_incidence;

    // phi of each crossing, for the whole batch ahead of the per-hit loop below, which does not vectorise.
    const MPlexQF h3_phi = Matriplex::atan2_poly(h3_state.y, h3_state.x);

    // The iteration's window factors, read once per batch.
    const float dphi_trk_fac = win_dphi_trk_fac(), dq_trk_fac = win_dq_trk_fac();

    // Post-process hits into the per-candidate queues. The cut is taken on
    // h3_state, the cubic solved onto the hit's module plane.
    for (int h = 0; h < N_proc_hits; ++h) {
      CandIncoming& cand = *cand_ptrs[cand_idcs[h]];
      float q, ddq, phi, ddphi;
      if (m_rz_limits.m_is_barrel) {
        q = h3_state.z[h];
      } else {
        q = hipo(h3_state.x[h], h3_state.y[h]);
      }
      ddq = std::abs(q - L.hit_q(hit_idcs[h]));
      phi = h3_phi[h];
      ddphi = cdist(std::abs(phi - L.hit_phi(hit_idcs[h])));

      // q error referenced to this module's plane, see surface_referenced_dq().
      const float dq_trk =
          Window::surface_q
              ? surface_referenced_dq(
                    B.m_dq_track[cand_idcs[h]], TCE, cand_idcs[h], h3_state, h, module_norm, m_rz_limits.m_is_barrel)
              : B.m_dq_track[cand_idcs[h]];

      // The pre-selection cut. See doc/MkFinderV2p2-DesignNotes.md, "Search window".
      const float dq_cut = dq_trk_fac * dq_trk + Window::dq_hit_fac * L.hit_q_half_length(hit_idcs[h]);
      const float dphi_cut = Window::capped_dphi(
          dphi_trk_fac * B.m_dphi_track[cand_idcs[h]] +
          (Window::phi_per_hit ? Window::dphi_hit_fac * L.hit_phi_half_extent(hit_idcs[h]) : Window::dphi_flat_rad));
      const bool dqdphi_presel = ddq < dq_cut && ddphi < dphi_cut;

      // For every hit, not only those that pass: the debug printout shows it.
      h3_state.dalpha[h] =
          B.m_sp1.dalpha[cand_idcs[h]] + h3dop.m_T[h] * (B.m_sp2.dalpha[cand_idcs[h]] - B.m_sp1.dalpha[cand_idcs[h]]);

#ifdef DEBUG
      // clang-format off
        dprintf("     SelHit %6.3f %6.3f %6.4f %7.5f   %6.4f   %s [dq = %d, dphi = %d]\n",
                L.hit_q(hit_idcs[h]), L.hit_phi(hit_idcs[h]),
                ddq, ddphi, h3_state.dalpha[h], dqdphi_presel ? "PASS" : "REJECT",
                (int) (ddq < dq_cut), (int) (ddphi < dphi_cut));
        dprintf("       ddq=%.3f, dq_track=%.4f, hit_q_half_len=%.4f, dq_cut=%.4f\n",
                ddq, B.m_dq_track[cand_idcs[h]], L.hit_q_half_length(hit_idcs[h]), dq_cut);
      // clang-format on
#endif

#ifdef MKFIT_TRACE
      // h3_state is the state the cut above was taken on.
      TrHitMatch& hm = mp_event->tr_hitmatch(tr_hitmatch_ids[h]);
      fill_hitmatch_presel(hm,
                           h3_state,
                           h,
                           h3dop.m_T[h],
                           d3[h],
                           L,
                           hit_orig_idcs[h],
                           ddphi,
                           ddq,
                           L.hit_q_half_length(hit_idcs[h]),
                           dqdphi_presel);
#ifdef MKFIT_TRACE_PROP_COMPARE
      hm.kine_on_plane_cmp = statep2bivec3(h_plex, h);
#endif
      if (dqdphi_presel)
        V2P2_TR_LS_COUNT(tr_layersearch_ids[cand_idcs[h]], n_hits_presel);
#endif

      if (!dqdphi_presel)
        continue;
      if (cos2_inc[h] < min_cos2_inc)
        continue;

      // The reduction is PER SUB-LAYER, so each sensor keeps its own budget. A full
      // queue takes the hit only in place of its worst one.
      const int sl = is_sec_layer ? 1 : 0;
      const bool full = cand.m_pqueue[sl].size() >= mp_job->v2p2().max_presel_hits;
      if (full && !(ddphi < cand.pq_top(sl).ddphi))
        continue;
      if (full)
        cand.pq_pop(sl);
      CandIncoming::PassedHitMatch phm{ddphi, hit_orig_idcs[h], hit_idcs[h], L.layer_id(), {h3_state, h, is_plex, h}};
#ifdef MKFIT_TRACE
      phm.tr_hitmatch_id = tr_hitmatch_ids[h];
#endif
      cand.pq_push(sl, phm);
    }
  }

  //----------------------------------------------------------------------------
  // Phase 4 -- move the pqueue survivors into the per-candidate hit list.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::order_hits_along_path(LayerBatch& b) {
    const int N_proc = b.N_proc;
    CandIncoming** cand_ptrs = b.cand;
#ifdef MKFIT_TRACE
    int* tr_layersearch_ids = b.tr_layersearch_ids;
#endif

    // Copy both sub-layer queues into one list, then order it by path length. The
    // queues only bound the number of hits; their order is not used.
    for (int i = 0; i < N_proc; ++i) {
      CandIncoming& cand = *cand_ptrs[i];
#ifdef MKFIT_TRACE
      mp_event->tr_layersearch(tr_layersearch_ids[i]).n_hits_pqueue = cand.m_pqueue[0].size() + cand.m_pqueue[1].size();
#endif
      for (int sl = 0; sl < 2; ++sl) {
        for (const auto& e : cand.m_pqueue[sl])
          cand.m_layer_hits.push_back(e);
        cand.m_pqueue[sl].clear();
      }

      // Path order for the in-layer search: dalpha is monotone in path length
      // for one candidate, but it is a signed turning angle, so its sign goes
      // with the charge. m_sp1 is the near bounding surface and m_sp2 the far
      // one in the search direction, so dalpha times the sign of its change
      // from m_sp1 to m_sp2 grows along the path, in either search direction.
      // Once per candidate is enough. See doc/MkFinderV2p2-DesignNotes.md,
      // "Reduction and hit ordering".
      const float dir = b.B.m_sp2.dalpha[i] >= b.B.m_sp1.dalpha[i] ? 1.0f : -1.0f;
      std::sort(cand.m_layer_hits.begin(),
                cand.m_layer_hits.end(),
                [dir](const CandIncoming::PassedHitMatch& a, const CandIncoming::PassedHitMatch& b) {
                  const float ka = dir * a.mixed_state.dalpha, kb = dir * b.mixed_state.dalpha;
                  if (ka != kb)
                    return ka < kb;
                  // Equal keys: hits on one module, or on coplanar modules of one layer,
                  // share the plane crossing. Larger ddphi first, the order the
                  // pqueue drain gave and the results were validated with, then by
                  // hit: an order that does not depend on the pqueues'.
                  if (a.ddphi != b.ddphi)
                    return a.ddphi > b.ddphi;
                  return a.hit_orig_index < b.hit_orig_index;
                });
    }

#ifdef MKFIT_TRACE
    // Ranks by the pre-selection's ddphi, 1 = best: sub_rank within the hit's
    // sub-layer, full_rank across both. Counted rather than sorted, so
    // m_layer_hits keeps its path order.
    for (int i = 0; i < N_proc; ++i) {
      CandIncoming& cand = *cand_ptrs[i];
      for (const auto& e : cand.m_layer_hits) {
        int better = 0, better_sub = 0;
        for (const auto& o : cand.m_layer_hits) {
          better += (o.ddphi < e.ddphi);
          better_sub += (o.ddphi < e.ddphi && o.layer == e.layer);
        }
        TrHitMatch& hm = mp_event->tr_hitmatch(e.tr_hitmatch_id);
        hm.passed_pqueue = true;
        hm.sub_rank = better_sub + 1;
        hm.full_rank = better + 1;
      }
    }
#endif
  }

  //----------------------------------------------------------------------------
  // In-layer combinatorial search: grow the tree of paths through the layer
  // over the path-ordered m_layer_hits. expand_in_layer() queues the depth-0
  // items; from there it is a worklist, not recursion: each flush harvests
  // nodes into m_deep_queue, rk_pump() turns them into deeper items, and so on
  // to max_hits_per_layer. Items in a batch that is not yet full wait for a later
  // layer batch or rk_drain(). Nodes live in m_node_arena until their
  // CombCandidate is selected. See doc/MkFinderV2p2-DesignNotes.md,
  // "In-layer combinatorial search".
  //----------------------------------------------------------------------------

  void MkFinderV2p2::expand_in_layer(LayerBatch& b) {
    rk_setup(m_koa_d0, false);
    rk_setup(m_koa_deep, true);
    m_kalman_out.clear();
    for (int i = 0; i < b.N_proc; ++i) {
      CandIncoming& cand = *b.cand[i];
      if (Policy::use_wsr && cand.m_wsr.m_wsr == WSR_Outside)
        continue;
      const int nlh = (int)cand.m_layer_hits.size();
      for (int lh = 0; lh < nlh; ++lh)
        rk_add_item(m_koa_d0, cand, cand.m_layer_hits[lh], -1, lh);
      rk_pump();
    }
  }

  //----------------------------------------------------------------------------
  // Running Kalman (rk_): the Kalman batches of the in-layer search stay open
  // across layer batches and fire only when full. Each CombCandidate counts its
  // Kalman items in flight and its nodes still to be expanded, so it can be
  // selected as soon as both reach zero.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::rk_setup(KalmanOpArgs& koa, bool solve_plane) {
    koa.prop_config = &mp_job->m_trk_info.prop_config();
    koa.mp_out = &m_kalman_out;
    koa.m_solve_plane = solve_plane;
    koa.m_chi2_trk_fac = mp_job->v2p2().chi2_trk_fac;
#ifdef MKFIT_TRACE
    koa.mp_event = mp_event;
#endif
  }

  void MkFinderV2p2::rk_add_item(
      KalmanOpArgs& koa, CandIncoming& cand, const CandIncoming::PassedHitMatch& phm, int parent_idx, int hit_pos) {
    koa.item_begin(&cand, {(int)phm.hit_orig_index, phm.layer}, parent_idx, hit_pos);
    if (parent_idx < 0)
      koa.load_state_err_chg(phm.mixed_state, cand.tcand().state());
    else
      koa.load_state_err_chg(m_node_arena[parent_idx].m_state);
#ifdef MKFIT_TRACE
    koa.set_tr_hitmatch_id(phm.tr_hitmatch_id);
#endif
    load_hit_module(koa, phm);
    ++cand.mp_ccrep->m_n_inflight;
    if (koa.item_finished())
      rk_flush(koa);
  }

  void MkFinderV2p2::rk_flush(KalmanOpArgs& koa) {
    if (koa.N_filled == 0)
      return;
    V2P2_COUNT(n_kalman_calls);
    V2P2_COUNT_ADD(n_kalman_lanes, koa.N_filled);
    if (!koa.m_solve_plane) {
      V2P2_COUNT(n_kalman_calls_d0);
      V2P2_COUNT_ADD(n_kalman_lanes_d0, koa.N_filled);
      koa.compute_pars();
    }
    koa.do_kalman_stuff();
    koa.reset();

    m_ccrep_touched.clear();
    for (const auto& o : m_kalman_out)
      m_ccrep_touched.push_back(o.cand->mp_ccrep);
    harvest_nodes();
    for (const int ni : m_nodes_harvested) {
      const CandNode& n = m_node_arena[ni];
      if (n.m_feat.n_hits < mp_job->v2p2().max_hits_per_layer) {
        m_deep_queue.push_back(ni);
        ++n.m_cand->mp_ccrep->m_n_inflight;
      }
    }
    // Items retire after their children are queued, so a count never passes
    // through zero while work remains.
    for (CCandRep* c : m_ccrep_touched)
      if (--c->m_n_inflight == 0 && c->m_n_queued == 0)
        m_ccrep_ready.push_back(c);
  }

  void MkFinderV2p2::rk_expand_node(int ni) {
    // By index throughout: a flush inside this loop may grow the arena. The node
    // itself cannot be freed meanwhile, its expansion still counts as in flight.
    CandIncoming& cand = *m_node_arena[ni].m_cand;
    const int nlh = (int)cand.m_layer_hits.size();
    for (int lh = m_node_arena[ni].m_hit_pos + 1; lh < nlh; ++lh) {
      const CandIncoming::PassedHitMatch& phm = cand.m_layer_hits[lh];
      // A second hit from a module already on this path is a split cluster or
      // another track's hit, not an overlap: skip it. The walk back is at most
      // V2p2Params::max_hits_per_layer long.
      bool same_module = false;
      for (int ci = ni; ci >= 0 && !same_module; ci = m_node_arena[ci].m_parent_idx) {
        const CandNode& an = m_node_arena[ci];
        if (an.m_hot.layer != phm.layer)
          continue;
        const auto& La = mp_job->m_event_of_hits[an.m_hot.layer];
        same_module = La.refHit(an.m_hot.index).detIDinLayer() == La.refHit(phm.hit_orig_index).detIDinLayer();
      }
      if (same_module) {
        V2P2_COUNT(n_same_module_vetoed);
        continue;
      }
      rk_add_item(m_koa_deep, cand, phm, ni, lh);
    }
  }

  void MkFinderV2p2::rk_pump() {
    while (!m_deep_queue.empty()) {
      const int ni = m_deep_queue.back();
      m_deep_queue.pop_back();
      CCandRep* c = m_node_arena[ni].m_cand->mp_ccrep;
      rk_expand_node(ni);
      if (--c->m_n_inflight == 0 && c->m_n_queued == 0)
        m_ccrep_ready.push_back(c);
    }
  }

  void MkFinderV2p2::rk_drain() {
    rk_setup(m_koa_d0, false);
    rk_setup(m_koa_deep, true);
    while (m_koa_d0.N_filled > 0 || m_koa_deep.N_filled > 0 || !m_deep_queue.empty()) {
      rk_flush(m_koa_d0);
      rk_pump();
      rk_flush(m_koa_deep);
      rk_pump();
    }
    m_ccrep_ready.clear();  // end_layer() selects everything not yet selected
  }

  void MkFinderV2p2::load_hit_module(KalmanOpArgs& koa, const CandIncoming::PassedHitMatch& phm) const {
    const auto& L = mp_job->m_event_of_hits[phm.layer];
    const Hit& hit = L.refHit(phm.hit_orig_index);
    koa.load_hit_module(hit, L.layer_info().module_info(hit.detIDinLayer()));
  }

  void MkFinderV2p2::harvest_nodes() {
    m_nodes_harvested.clear();
    for (const auto& o : m_kalman_out) {
      // Also rejects NaN. A negative chi2 means the predicted covariance is broken
      // (design notes, "Grazing incidence"); the score would reward it. So does det V <= 0:
      // the residual covariance is then indefinite, chi2 can still come out >= 0 along the
      // residual, and ln(det V), floored below, would give the hit a large bonus.
      if (!(o.chi2_acc < mp_job->v2p2().hit_chi2_cut && o.chi2_acc >= 0.0f && o.chi2 >= 0.0f && o.det_v > 0.0f))
        continue;
      CandNode n;
      n.m_cand = o.cand;
      n.m_parent_idx = o.parent_idx;
      n.m_hit_pos = o.hit_pos;
      n.m_hot = o.hot;
      n.m_state = o.state;
      n.m_chi2 = o.chi2;

      // Features accumulate parent -> child. The parent is always already in the
      // arena: a node's children are queued only after it has been harvested.
      const LayerStepFeatures* pf = (o.parent_idx >= 0) ? &m_node_arena[o.parent_idx].m_feat : nullptr;
      LayerStepFeatures& f = n.m_feat;
      if (pf)
        f = *pf;
      else
        fill_step_geometry(f, o.cand->m_log_rho);
      if (pf)
        V2P2_COUNT(n_nodes_deep);
      f.n_hits = (pf ? pf->n_hits : 0) + 1;
      f.n_overlap = f.n_hits - 1;
      f.hole_kind = V2P2_NoHole;
      f.chi2_sum = (pf ? pf->chi2_sum : 0.0f) + o.chi2;
      f.log_det_v_sum = (pf ? pf->log_det_v_sum : 0.0f) + std::log(std::max(1e-30f, o.det_v));
#ifdef MKFIT_TRACE
      n.m_tr_hitmatch_id = o.tr_hitmatch_id;
#endif
      const int slot = node_slot();  // may grow the arena: pf is dead by now
      m_node_arena[slot] = n;        // a reused slot is overwritten whole
      o.cand->mp_ccrep->m_nodes.push_back(slot);
      m_nodes_harvested.push_back(slot);
    }
    m_kalman_out.clear();
    V2P2_COUNT_ADD(n_nodes, (long)m_nodes_harvested.size());
  }

  int MkFinderV2p2::node_slot() {
    if (m_node_free.empty()) {
      m_node_arena.emplace_back();
      return (int)m_node_arena.size() - 1;
    }
    const int slot = m_node_free.back();
    m_node_free.pop_back();
    return slot;
  }

  // The parts of a layer step that do not depend on which hits were taken: where
  // the step goes and which way. Filled once per path root and carried down the
  // tree, and once per hole entry of the end-of-layer selection.
  void MkFinderV2p2::fill_step_geometry(LayerStepFeatures& f, float log_rho) const {
    const LayerInfo& li = m_rz_limits.layer_info_1();
    f.layer_to = (short)li.layer_id();
    f.is_outward = m_rz_limits.is_outward();
    f.is_pixel = li.is_pixel();
    f.is_barrel = li.is_barrel();
    f.log_rho = log_rho;
  }

  //----------------------------------------------------------------------------
  // End-of-layer selection: every in-layer path, every candidate as the hole it
  // would record, and every candidate that did not enter the layer compete in
  // one sort on the additive layer-step score; the top CCandRep::m_max_cands
  // survive.
  // See doc/MkFinderV2p2-DesignNotes.md, "End-of-layer selection".
  //----------------------------------------------------------------------------

  void MkFinderV2p2::select_completed_ccreps(const LayerBatch& b) {
    // The batch's incoming candidates stay alive: end_layer() clears m_cands.
    auto try_select = [this](CCandRep& ccrep) {
      if (ccrep.m_n_queued > 0 || ccrep.m_n_inflight > 0 || ccrep.m_selected)
        return;
      select_and_materialise(ccrep);
      ccrep.m_selected = true;
      for (const int ni : ccrep.m_nodes)
        m_node_free.push_back(ni);
      ccrep.m_nodes.clear();
    };
    for (int i = 0; i < b.N_proc; ++i)
      try_select(*b.cand[i]->mp_ccrep);
    for (CCandRep* c : m_ccrep_ready)
      try_select(*c);
    m_ccrep_ready.clear();
  }

  void MkFinderV2p2::select_and_materialise(CCandRep& ccrep) {
    CombCandidate& ccand = ccrep.m_ccand;
    const int max_cands = ccrep.m_max_cands > 0 ? ccrep.m_max_cands : ccand.capacity();

    m_sel.clear();

    // Which TrackCands got a CandIncoming this layer. The rest are entered
    // unchanged below.
    const int n_tc = (int)ccand.size();
    m_has_cand.assign(n_tc, 0);

    for (auto& cand : ccrep.m_cands) {
      m_has_cand[cand.m_origin_tcand_index] = 1;

      TrackCand& tc = cand.tcand();

      // Every CandIncoming also competes as the hole it would record, including
      // those that found paths. In the log-likelihood score (Score::mode 1) a hole
      // adds exactly 0, whatever its kind; hole penalties apply in the final pick.
      // WSR_Outside is not a hole: the candidate competes unchanged, unless
      // V2p2Params::outside_hole_max_seed_hits makes it one for a short seed.
      const int outside_max_seed = mp_job->v2p2().outside_hole_max_seed_hits;
      const bool outside_as_hole =
          outside_max_seed > 0 && tc.getNSeedHits() <= outside_max_seed && tc.nFoundHits() <= tc.getNSeedHits();
      if (Policy::use_wsr && cand.m_wsr.m_wsr == WSR_Outside && !outside_as_hole) {
        m_sel.push_back({cand.m_origin_tcand_index, -1, 0, false, tc.score()});
      } else {
        const int fake = fake_hit_index(tc, cand.m_wsr);
        LayerStepFeatures f;
        fill_step_geometry(f, 0.0f);
        f.hole_kind = v2p2_hole_kind(fake);
        m_sel.push_back(
            {cand.m_origin_tcand_index, -1, fake, true, tc.score() + v2p2_layer_step_score(f, mp_job->v2p2())});
      }
    }

    for (const int ni : ccrep.m_nodes) {
      const CandNode& n = m_node_arena[ni];
      m_sel.push_back({n.m_cand->m_origin_tcand_index,
                       ni,
                       0,
                       false,
                       n.m_cand->tcand().score() + v2p2_layer_step_score(n.m_feat, mp_job->v2p2())});
    }

    // Stopped or pull-in-skipped TrackCands, unchanged, at their own score. With
    // best-short on (outward only), stopped ones leave the beam instead and the
    // best is kept on the CombCandidate.
    const bool best_short = InLayer::best_short && m_rz_limits.is_outward();
    for (int ic = 0; ic < n_tc; ++ic) {
      if (m_has_cand[ic])
        continue;
      if (best_short && ccand[ic].getLastHitIdx() == Hit::kHitStopIdx) {
        offer_best_short(ccand, ccand[ic]);
        continue;
      }
      m_sel.push_back({ic, -1, 0, false, ccand[ic].score()});
    }

    if (m_sel.empty()) {
      ccand.clear();  // every TrackCand was offered to best-short and left the beam
      return;
    }

    if (InLayer::sister_hole_drop)
      drop_sister_dominated_holes(ccand);

    // Partial sort: only the top max_cands matter.
    const int n_keep = std::min((int)m_sel.size(), max_cands);
    std::partial_sort(m_sel.begin(), m_sel.begin() + n_keep, m_sel.end(), [](const SelEntry& a, const SelEntry& b) {
      return a.score > b.score;
    });

    // Reserve a hole slot: keep the best candidate that declined the layer, even
    // if outranked, so one branch stays open to an earlier hit being wrong.
    if (InLayer::reserve_hole_slot && n_keep > 1) {
      bool kept_a_decliner = false;
      for (int k = 0; k < n_keep && !kept_a_decliner; ++k)
        kept_a_decliner = (m_sel[k].node_idx < 0);
      if (!kept_a_decliner) {
        int best_decliner = -1;
        for (int k = n_keep; k < (int)m_sel.size(); ++k)
          if (m_sel[k].node_idx < 0 && (best_decliner < 0 || m_sel[k].score > m_sel[best_decliner].score))
            best_decliner = k;
        if (best_decliner >= 0) {
          std::swap(m_sel[n_keep - 1], m_sel[best_decliner]);
          V2P2_COUNT(n_hole_slot_reserved);
        }
      }
    }

    // Build the survivors as copies BEFORE touching the CombCandidate: several
    // winners can descend from the same TrackCand -- that is what branching is --
    // so the sources have to stay readable while the copies are made.
    m_new_cands.clear();
    for (int k = 0; k < n_keep; ++k) {
      const SelEntry& e = m_sel[k];
      m_new_cands.push_back(ccand[e.tcand_idx]);
      TrackCand& nc = m_new_cands.back();

      if (e.node_idx >= 0) {
        int chain[InLayer::max_hits_per_layer_limit];
        int n_chain = 0;
        for (int ci = e.node_idx; ci >= 0; ci = m_node_arena[ci].m_parent_idx)
          chain[n_chain++] = ci;
        for (int c = n_chain - 1; c >= 0; --c) {
          const CandNode& n = m_node_arena[chain[c]];
          nc.addHitIdx(n.m_hot.index, n.m_hot.layer, n.m_chi2);
          if (c != n_chain - 1) {
            nc.incOverlapCount();  // a second hit in one layer: an overlap, or the sister sensor of a paired entry
          }
#ifdef MKFIT_TRACE
          int pid = nc.m_trace_state_id;
          int id = mp_event->trace_new_cand_state(pid, (*mp_steeringparams_iter)->m_layer, track2bivec3(nc), n.m_state);
          if (n.m_tr_hitmatch_id >= 0) {
            auto& ku = mp_event->tr_kalmanupdate(mp_event->tr_hitmatch(n.m_tr_hitmatch_id).kalman_id);
            ku.accepted = true;
            ku.state_id_out = id;
          }
          nc.m_trace_state_id = id;
#endif
        }
        nc.setState(m_node_arena[e.node_idx].m_state);
        V2P2_COUNT(n_path_taken);
        if (n_chain > 1)
          V2P2_COUNT_ADD(n_extra_hits, n_chain - 1);
      } else if (e.add_fake) {
        switch (e.fake_hit) {
          case Hit::kHitMissIdx:
            V2P2_COUNT(n_hole);
            break;
          case Hit::kHitEdgeIdx:
            V2P2_COUNT(n_hot_edge);
            break;
          case Hit::kHitInGapIdx:
            V2P2_COUNT(n_hot_gap);
            break;
          case Hit::kHitStopIdx:
            V2P2_COUNT(n_stop_holes);
            break;
          default:
            break;
        }
        if (e.fake_hit == Hit::kHitStopIdx)
          nc.setStopReason(TrackCand::SR_TooManyHoles);
        nc.addHitIdx(e.fake_hit, m_rz_limits.layer_info_1().layer_id(), 0.0f);
#ifdef MKFIT_TRACE
        int pid = nc.m_trace_state_id;
        nc.m_trace_state_id =
            mp_event->trace_new_cand_state(pid, (*mp_steeringparams_iter)->m_layer, track2bivec3(nc), nc.state());
#endif
      }

      // score_ holds the summed layer-step score during finding; the final-pick
      // scorer overwrites it in mergeCandsAndBestShortOne() at the end of the search.
      nc.setScore(e.score);
#ifdef MKFIT_TRACE
      nc.note_rank(k + 1, m_rz_limits.layer_info_1().layer_id(), Diag::rank_over);
#endif
    }

    // Candidates that stopped IN this layer leave the beam the same way.
    ccand.clear();
    for (auto& c : m_new_cands) {
      if (best_short && c.getLastHitIdx() == Hit::kHitStopIdx) {
        offer_best_short(ccand, c);
        continue;
      }
      ccand.push_back(c);
    }

    V2P2_COUNT(n_selections);
    V2P2_COUNT_ADD(n_sel_entries, (long)m_sel.size());
    V2P2_COUNT_ADD(n_sel_kept, n_keep);
  }

  // InLayer::sister_hole_drop: remove from m_sel the holes that a path of the same
  // TrackCand dominates by taking the sister sensor of its last hit. A sister pair
  // is a double layer (TrackerInfo::get_double_layers()), and the two sensors of one
  // module have detids one apart; either may be crossed first. A
  // track through one sensor of a module crosses the other a few mm away, so
  // when a path took the sister hit, the hole is the worse explanation of the
  // same crossing. The first hit of the path is the one tested. Written for the
  // split plan, where the previous hit can be in the sister plan entry; for a
  // paired entry it drops nothing.
  void MkFinderV2p2::drop_sister_dominated_holes(const CombCandidate& ccand) {
    const int lay = m_rz_limits.layer_info_1().layer_id();
    const TrackerInfo& ti = mp_job->m_trk_info;
    auto detid_of = [this](int layer, int idx) {
      const LayerOfHits& L = mp_job->m_event_of_hits[layer];
      return L.layer_info().module_info(L.refHit(idx).detIDinLayer()).detid;
    };
    int w = 0;
    for (int k = 0; k < (int)m_sel.size(); ++k) {
      const SelEntry e = m_sel[k];
      bool drop = false;
      if (e.node_idx < 0 && e.add_fake) {
        const TrackCand& tc = ccand[e.tcand_idx];
        const int pl = tc.getLastHitLyr(), pi = tc.getLastHitIdx();
        if (pi >= 0 && ti.layer(lay).double_layer_partner() == pl) {
          const unsigned int d_prev = detid_of(pl, pi);
          for (const SelEntry& f : m_sel) {
            if (f.node_idx < 0 || f.tcand_idx != e.tcand_idx)
              continue;
            int root = f.node_idx;
            while (m_node_arena[root].m_parent_idx >= 0)
              root = m_node_arena[root].m_parent_idx;
            const CandNode& r = m_node_arena[root];
            if (InLayer::sister_hole_chi2 > 0.0f && !(r.m_chi2 < InLayer::sister_hole_chi2))
              continue;
            const unsigned int d_this = detid_of(r.m_hot.layer, r.m_hot.index);
            if (d_this == d_prev + 1 || d_prev == d_this + 1) {
              drop = true;
              break;
            }
          }
        }
      }
      if (drop)
        V2P2_COUNT(n_sister_hole_dropped);
      else
        m_sel[w++] = e;
    }
    m_sel.resize(w);
  }

  // Keep the best-scoring stopped candidate of this CombCandidate. The score is
  // the in-flight accumulator, which is on one scale within a seed -- the only
  // comparison made here.
  void MkFinderV2p2::offer_best_short(CombCandidate& ccand, const TrackCand& tc) const {
    V2P2_COUNT(n_best_short_offered);
    if (ccand.refBestShortCand().combCandidate() == nullptr || tc.score() > ccand.refBestShortCand().score()) {
      ccand.setBestShortCand(tc);
      V2P2_COUNT(n_best_short_taken);
    }
  }

}  // namespace mkfit
