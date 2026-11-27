// The backward fit of MkFinderV2p2 tracks. A member of MkFinder, as it runs on
// MkFinder's Matriplex state and on bkFitInputTracks() / bkFitOutputTracks(),
// but kept here so that MkFinder::bkFitFitTracksProp2Plane(), used by V1 / V2,
// stays as it is upstream. MkBuilder::fit_cands() picks this one when the
// candidates come from MkBuilder::findTracksStandardv2p2().
//
// It walks the hits on track backwards with propagation to each hit's module
// plane and needs no layer plan.

#include "MkFinder.h"

#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "KalmanUtilsMPlex.h"

#include <algorithm>

//#define DEBUG
#include "Debug.h"

#if defined(MKFIT_STANDALONE)
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

#ifdef MKFIT_TRACE
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntConversions.h"
#endif

namespace mkfit {

#ifdef MKFIT_TRACE
  namespace {
    // Trace of the backward fit: one TrBkFitUpdate per fitted hit and one
    // TrCandState chain per lane, advanced at every hit. The chain is kept here
    // rather than in TrackCand::m_trace_state_id, which belongs to the search:
    // writing it would chain the search's states onto the fit's.
    struct BkFitTracer {
      const Event *m_event;
      int m_layer[NN], m_mcid[NN], m_hit[NN], m_step[NN], m_state[NN];
      bool m_live[NN];

      // The root state of each lane is the fit's INPUT state, after
      // bkFitInputTracks() has inflated the covariance by 100 (10x in sigma).
      BkFitTracer(const Event *ev,
                  TrackCand *const *tcands,
                  const MPlexLV &par,
                  const MPlexLS &err,
                  const MPlexQI &chg,
                  int N_proc)
          : m_event(ev) {
        std::fill_n(m_layer, NN, -1);
        std::fill_n(m_mcid, NN, -1);
        std::fill_n(m_hit, NN, -1);
        std::fill_n(m_step, NN, 0);
        std::fill_n(m_state, NN, -1);
        std::fill_n(m_live, NN, false);
        if (!m_event)
          return;
        for (int i = 0; i < N_proc; ++i) {
          TrackCand *tc = tcands[i];
          CombCandidate *cc = tc ? tc->combCandidate() : nullptr;
          if (cc == nullptr)
            continue;
          // Created lazily, as in MkBuilder: in HLT the fit runs before any search.
          if (cc->m_trace_meta_id == -1)
            cc->m_trace_meta_id = m_event->trace_new_cand_meta(m_event->evtID(), cc->seed_origin_index());
          TrackState ts;
          par.copyOut(i, ts.parArray_nc());
          err.copyOut(i, ts.errArray_nc());
          ts.charge = chg[i];
          // layer -1: the root sits before any hit. stage 1 = BkwFit.
          auto [stage_id, state_id] =
              m_event->trace_new_cand_stage_and_state(cc->m_trace_meta_id, -1, 1, -1, state2bivec3(ts), ts);
          cc->m_trace_stage_id = stage_id;
          TrCandMeta &cm = m_event->tr_candmeta(cc->m_trace_meta_id);
          cm.stage_ids[1] = stage_id;
          // The truth-join key, if MkBuilder has not set it (HLT runs the fit first):
          // seeds are relabelled sequentially, so a track's label is its seed index.
          if (cm.global_seed == -1 && tc->label() >= 0)
            cm.global_seed = tc->label();
          m_state[i] = state_id;
        }
      }

      void on_hit(int i, int layer, int hit_idx, int mc_hit_id) {
        if (!m_event)
          return;
        m_layer[i] = layer;
        m_hit[i] = hit_idx;
        m_mcid[i] = mc_hit_id;
        m_live[i] = true;
      }

      // After the update: par_p / err_p propagated, par_c / err_c updated.
      void on_update(int N_proc,
                     const MPlexLV &par_p,
                     const MPlexLS &err_p,
                     const MPlexLV &par_c,
                     const MPlexLS &err_c,
                     const MPlexQI &chg,
                     const MPlexHV &ms_par,
                     const MPlexHV &pl_nrm,
                     const MPlexHV &pl_dir,
                     const MPlexQF &chi2,
                     const MPlexQF &chi2_cum,
                     const MPlexQI &fail) {
        if (!m_event)
          return;
        for (int i = 0; i < N_proc; ++i) {
          if (!m_live[i])
            continue;
          m_live[i] = false;
          const int step = m_step[i]++;
          if (m_state[i] < 0)
            continue;

          TrBkFitUpdate bu;
          bu.state_id_in = m_state[i];
          bu.layer = m_layer[i];
          bu.hit = m_hit[i];
          bu.step = step;
          bu.fail = fail[i];
          bu.chi2 = chi2.constAt(i, 0, 0);
          bu.chi2_cum = chi2_cum.constAt(i, 0, 0);

          // The PROPAGATED state on the plane, and its residual against the hit in
          // the module frame. ydir = zdir x xdir.
          const EVec3 pos(par_p.constAt(i, 0, 0), par_p.constAt(i, 1, 0), par_p.constAt(i, 2, 0));
          const EVec3 d = pos - EVec3(ms_par.constAt(i, 0, 0), ms_par.constAt(i, 1, 0), ms_par.constAt(i, 2, 0));
          const EVec3 n(pl_nrm.constAt(i, 0, 0), pl_nrm.constAt(i, 1, 0), pl_nrm.constAt(i, 2, 0));
          const EVec3 u(pl_dir.constAt(i, 0, 0), pl_dir.constAt(i, 1, 0), pl_dir.constAt(i, 2, 0));
          const float ipt = par_p.constAt(i, 3, 0);
          const float pt = ipt != 0.f ? 1.f / std::abs(ipt) : 0.f;
          const float phi = par_p.constAt(i, 4, 0);
          const float theta = par_p.constAt(i, 5, 0);
          bu.kine_on_plane = {pos, EVec3(pt * std::cos(phi), pt * std::sin(phi), pt / std::tan(theta))};
          bu.residual_x = d.Dot(u);
          bu.residual_y = d.Dot(n.Cross(u));
          bu.residual_z = d.Dot(n);

          // Which particle made the hit. Not compared to anything here: the track's
          // sim label is not available at this point (see the struct).
          if (m_mcid[i] >= 0 && m_mcid[i] < (int)m_event->simHitsInfo_.size())
            bu.mc_track_id = m_event->simHitsInfo_[m_mcid[i]].mcTrackID();

          TrackState ts;
          par_c.copyOut(i, ts.parArray_nc());
          err_c.copyOut(i, ts.errArray_nc());
          ts.charge = chg[i];
#ifdef MKFIT_TRACE_KALMAN_DEBUG
          par_p.copyOut(i, bu.propagated_state.parArray_nc());
          err_p.copyOut(i, bu.propagated_state.errArray_nc());
          bu.propagated_state.charge = chg[i];
#endif
          const int sid = m_event->trace_new_cand_state(m_state[i], m_layer[i], state2bivec3(ts), ts);
          bu.state_id_out = sid;
          m_state[i] = sid;
          m_event->trace_bkfitupdate(std::move(bu));
        }
      }
    };
  }  // namespace
#endif

  void MkFinder::bkFitFitTracksV2p2(const EventOfHits &eventofhits,
                                    const SteeringParams &st_par,
                                    const int N_proc,
                                    bool chiDebug) {
    // Works with track-finding hit indices, before remapping.

    // bool debug = true;

    MPlexQF tmp_chi2{0.0f};
    MPlexQI done_flag(0);
    // The last hit fitted per lane, to skip a repeated hit on track (seed-hit
    // merging can put the same hit on twice).
    int fit_lay[NN], fit_idx[NN];
    for (int i = 0; i < NN; ++i)
      fit_lay[i] = fit_idx[i] = -1;

#ifdef MKFIT_TRACE
    BkFitTracer tracer(m_event, m_TrkCand, m_Par[iC], m_Err[iC], m_Chg, N_proc);
#endif

    MPlexHV plNrm{0.0f};  // input detector plane [pl - plane]
    MPlexHV plDir{0.0f};  // ""
    MPlexHV plPnt{0.0f};  // ""

    // Computed at full width, see pad_free_lanes(); only the first N_proc lanes are read out.
    pad_free_lanes(N_proc, m_Par[iC], m_Err[iC], m_Chg);

#if defined(DEBUG_PROP_UPDATE)
    const int DSLOT = 0;
    int DSLOT_layer;
    printf("bkfit-p2p entry, track in slot %d\n", DSLOT);
    print_par_err(iC, DSLOT);
#endif
#if defined(DEBUG_BACKWARD_FIT)
    const Hit *last_hit_ptr[NN];
    int last_layer[NN];
#endif

    // Outlier rejection, as MkFinder::bkFitFitTracksProp2Plane(): the IterationConfig's
    // m_backward_fit_outlier_* (chi2 0 = off).
    const float outlier_chi2 = m_iteration_config->m_backward_fit_outlier_chi2;
    const int max_outliers = m_iteration_config->m_backward_fit_max_outliers;
    const float outlier_min_pt = m_iteration_config->m_backward_fit_outlier_min_pt;
    int n_outliers[NN] = {0};

    // Path order within a physical layer (TrackerInfo::get_double_layers()). The HoT chain holds
    // the hits of the two sensor layers of a module pair in plan order, and seed hits on one
    // layer in the seed's order; neither is always the order the track meets them. So when a
    // lane comes to a physical layer it takes all its hits on that layer, a run, and fits them
    // nearest module plane first, by the straight-line path from its state at the start of the
    // run. Holes and repeated hits are skipped as before. The HoT chain is not changed.
    constexpr int kMaxRun = 16;  // a longer run is split
    int run_node[NN][kMaxRun];
    int run_n[NN] = {0}, run_pos[NN] = {0};

    const TrackerInfo &trk_info = *m_prop_config->backward_fit_pflags.tracker_info;
    auto collect_run = [&](int i) {
      run_n[i] = run_pos[i] = 0;
      std::pair<int, int> dl{-1, -1};
      for (; m_CurNode[i] >= 0; m_CurNode[i] = m_HoTNodeArr[i][m_CurNode[i]].m_prev_idx) {
        const HitOnTrack &hot = m_HoTNodeArr[i][m_CurNode[i]].m_hot;
        if (hot.index < 0)
          continue;
        // A repeat of the hit just fitted or of one in the run (seed-hit merging can put a
        // hit on twice).
        bool repeat = hot.layer == fit_lay[i] && hot.index == fit_idx[i];
        for (int k = 0; k < run_n[i] && !repeat; ++k) {
          const HitOnTrack &h = m_HoTNodeArr[i][run_node[i][k]].m_hot;
          repeat = h.layer == hot.layer && h.index == hot.index;
        }
        if (repeat)
          continue;
        if (dl.first < 0)
          dl = trk_info.get_double_layers(hot.layer);
        else if ((hot.layer != dl.first && hot.layer != dl.second) || run_n[i] == kMaxRun)
          break;
        run_node[i][run_n[i]++] = m_CurNode[i];
      }
      if (run_n[i] < 2)
        return;

      // s = (plane point - x) . n / (p^ . n): positive ahead along the momentum, negative behind.
      // The fit walks backwards, so the largest s comes first; equal s keep the chain order.
      const float x = m_Par[iC](i, 0, 0), y = m_Par[iC](i, 1, 0), z = m_Par[iC](i, 2, 0);
      const float phi = m_Par[iC](i, 4, 0), theta = m_Par[iC](i, 5, 0);
      const float px = std::cos(phi) * std::sin(theta), py = std::sin(phi) * std::sin(theta), pz = std::cos(theta);
      float s[kMaxRun];
      for (int k = 0; k < run_n[i]; ++k) {
        const HitOnTrack &hot = m_HoTNodeArr[i][run_node[i][k]].m_hot;
        const LayerOfHits &L = eventofhits[hot.layer];
        const ModuleInfo &mi = L.layer_info().module_info(L.refHit(hot.index).detIDinLayer());
        const float nx = mi.zdir[0], ny = mi.zdir[1], nz = mi.zdir[2];
        s[k] = ((mi.pos[0] - x) * nx + (mi.pos[1] - y) * ny + (mi.pos[2] - z) * nz) / (px * nx + py * ny + pz * nz);
      }
      for (int k = 1; k < run_n[i]; ++k) {
        for (int j = k; j > 0 && s[j] > s[j - 1]; --j) {
          std::swap(s[j], s[j - 1]);
          std::swap(run_node[i][j], run_node[i][j - 1]);
        }
      }
    };

    int done_count = 0;
    while (done_count != N_proc) {
      int fit_node[NN];  // HoT node fitted in this step, -1 if none
      std::fill_n(fit_node, NN, -1);
      int here_count = 0;
      for (int i = 0; i < N_proc; ++i) {
        if (done_flag[i])
          continue;

        if (run_pos[i] == run_n[i])
          collect_run(i);

        if (run_pos[i] == run_n[i]) {
          // Mark as done and copy out.
          done_flag[i] = 1;
          ++done_count;

          TrackCand &trk = *m_TrkCand[i];
          m_Err[iC].copyOut(i, trk.errors_nc().Array());
          m_Par[iC].copyOut(i, trk.parameters_nc().Array());
          trk.setCharge(m_Chg[i]);
          trk.setChi2(m_Chi2[i]);
          // Outliers became misses: the hole and overlap counts follow before the score.
          if (n_outliers[i] > 0)
            trk.recountHolesAndOverlaps();
          if (isFinite(trk.chi2())) {
            trk.setScore(getScoreCand(m_steering_params->m_post_bkfit_track_scorer, trk));
          }
        } else {
          // Prepare the next hit of the run and its module info. Every hit is fitted,
          // overlap and sister hits included. (The V1 / V2 fit keeps only the first of
          // consecutive hits in a layer.)
          const int node = run_node[i][run_pos[i]++];
          const int layer = m_HoTNodeArr[i][node].m_hot.layer;
          fit_lay[i] = layer;
          fit_idx[i] = m_HoTNodeArr[i][node].m_hot.index;

          const LayerOfHits &L = eventofhits[layer];
          const Hit &hit = L.refHit(m_HoTNodeArr[i][node].m_hot.index);
          const ModuleInfo &mi = L.layer_info().module_info(hit.detIDinLayer());

          m_msErr.copyIn(i, hit.errArray());
          m_msPar.copyIn(i, hit.posArray());
          plNrm.copyIn(i, mi.zdir.Array());
          plDir.copyIn(i, mi.xdir.Array());
          plPnt.copyIn(i, mi.pos.Array());

#ifdef MKFIT_TRACE
          tracer.on_hit(i, layer, m_HoTNodeArr[i][node].m_hot.index, hit.mcHitID());
#endif

          fit_node[i] = node;
          ++here_count;

#ifdef DEBUG_BACKWARD_FIT
          last_hit_ptr[i] = &hit;
          last_layer[i] = layer;
#endif
#if defined(DEBUG_PROP_UPDATE)
          DSLOT_layer = layer;
          // clang-format off
          printf("\nbkfit start layer %d, track in slot %d -- fail=%d, hit_xyz = (%g, %g, %g)\n\n",
                 DSLOT_layer, DSLOT, m_FailFlag[DSLOT],
                 m_msPar(DSLOT, 0, 0), m_msPar(DSLOT, 1, 0), m_msPar(DSLOT, 2, 0));
          // clang-format on
#endif
        }
      }

      if (done_count == N_proc)
        break;
      if (here_count == 0)
        continue;

      pad_free_lanes(N_proc, m_msErr, m_msPar, plNrm, plDir, plPnt);

      clearFailFlag();

      // PROP-FAIL-ENABLE We do not check for pfailed propagation here
      // clang-format off

      m_FailFlag.setVal(0);
      const MPlexQI chg_prev = m_Chg;
      propagateHelixToPlaneMPlex(m_Err[iC], m_Par[iC], m_Chg, plPnt, plNrm,
                                 m_Err[iP], m_Par[iP], m_FailFlag,
                                 NN, m_prop_config->backward_fit_pflags, nullptr);
      kalmanOperationPlaneLocal(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                                m_Err[iP], m_Par[iP], m_Chg, m_msErr, m_msPar, plNrm, plDir, plPnt,
                                m_Err[iC], m_Par[iC], tmp_chi2, NN);
      kalmanCheckChargeFlip(m_Par[iC], m_Chg, NN);

      // Outlier rejection: the hit becomes a missing hit and
      // the propagated state is kept. The negated comparison leaves a NaN chi2 alone.
      if (outlier_chi2 > 0.0f) {
        for (int i = 0; i < N_proc; ++i) {
          if (fit_node[i] < 0 || !(tmp_chi2[i] > outlier_chi2) ||
              n_outliers[i] >= max_outliers || m_TrkCand[i]->pT() < outlier_min_pt)
            continue;
          m_Err[iC].copySlot(i, m_Err[iP]);
          m_Par[iC].copySlot(i, m_Par[iP]);
          m_Chg[i] = chg_prev[i];
          tmp_chi2[i] = 0.0f;
          TrackCand &trk = *m_TrkCand[i];
          trk.combCandidate()->hot_node_nc(fit_node[i]).m_hot.index = Hit::kHitMissIdx;
          trk.setNFoundHits(trk.nFoundHits() - 1);
          trk.setNMissingHits(trk.nMissingHits() + 1);
          ++n_outliers[i];
        }
      }

#ifdef MKFIT_TRACE
      tracer.on_update(N_proc, m_Par[iP], m_Err[iP], m_Par[iC], m_Err[iC], m_Chg, m_msPar, plNrm, plDir,
                       tmp_chi2, m_Chi2, m_FailFlag);
#endif

#if defined(DEBUG_PROP_UPDATE)
      printf("\nbkfit at layer %d, track in slot %d -- fail=%d, hit_xyz = (%g, %g, %g)\n",
             DSLOT_layer, DSLOT, m_FailFlag[DSLOT],
             m_msPar(DSLOT, 0, 0), m_msPar(DSLOT, 1, 0), m_msPar(DSLOT, 2, 0));
      printf("Propagated:\n");
      print_par_err(iP, DSLOT);
      printf("Updated:\n");
      print_par_err(iC, DSLOT);
#endif
      // clang-format on

#if defined(DEBUG_BACKWARD_FIT)
      // clang-format off
      bool debug = true;
      const char beg_cur_sep = '/'; // set to ' ' root parsable printouts
      for (int i = 0; i < N_proc; ++i) {
        if (chiDebug && last_hit_ptr[i]) {
          TrackCand &bb = *m_TrkCand[i];
          int ti = iP;
          float chi = tmp_chi2.At(i, 0, 0);
          float chi_prnt = std::isfinite(chi) ? chi : -9;
          const int layer = last_layer[i];
          const LayerOfHits &L = eventofhits[layer];

#if defined(MKFIT_STANDALONE)
          const MCHitInfo &mchi = m_event->simHitsInfo_[last_hit_ptr[i]->mcHitID()];

          dprintf("BKF_OVERLAP %d %d %d %d %d %d %d "
                  "%f%c%f %f %f%c%f %f %f %f %d %d %d %d "
                  "%f %f %f %f %f\n",
              m_event->evtID(),
#else
          dprintf("BKF_OVERLAP %d %d %d %d %d %d "
                  "%f%c%f %f %f%c%f %f %f %f %d %d %d "
                  "%f %f %f %f %f\n",
#endif
              bb.label(), (int)bb.prodType(), bb.isFindable(),
              layer, L.is_stereo(), L.is_barrel(),
              bb.pT(), beg_cur_sep, 1.0f / m_Par[ti].At(i, 3, 0),
              bb.posEta(),
              bb.posPhi(), beg_cur_sep, std::atan2(m_Par[ti].At(i, 1, 0), m_Par[ti].At(i, 0, 0)),
              hipo(m_Par[ti].At(i, 0, 0), m_Par[ti].At(i, 1, 0)),
              m_Par[ti].At(i, 2, 0),
              chi_prnt,
              std::isnan(chi), std::isfinite(chi), chi > 0,
#if defined(MKFIT_STANDALONE)
              mchi.mcTrackID(),
#endif
              // The following three can get negative / prouce nans in e2s.
              // std::abs the args for FPE hunt.
              e2s(std::abs(m_Err[ti].At(i, 0, 0))),
              e2s(std::abs(m_Err[ti].At(i, 1, 1))),
              e2s(std::abs(m_Err[ti].At(i, 2, 2))),  // sx_t sy_t sz_t -- track errors
              1e4f * hipo(m_msPar.At(i, 0, 0) - m_Par[ti].At(i, 0, 0),
                                m_msPar.At(i, 1, 0) - m_Par[ti].At(i, 1, 0)),  // d_xy
              1e4f * (m_msPar.At(i, 2, 0) - m_Par[ti].At(i, 2, 0))             // d_z
          );
        }
      }
      // clang-format on
#endif

      // update chi2
      m_Chi2.add(tmp_chi2);
    }
  }

}  // namespace mkfit
