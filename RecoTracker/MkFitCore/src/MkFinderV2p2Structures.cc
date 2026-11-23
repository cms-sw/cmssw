#include "MkFinderV2p2Structures.h"
#include "MkFinderV2p2.h"
#include "PropagationMPlex.h"
#include "KalmanUtilsMPlex.h"
#include "PlaneSteps.h"
#include "MatriplexPackers.h"

#ifdef MKFIT_TRACE
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

namespace mkfit {

  void KalmanOpArgs::do_kalman_stuff() {
    // Propagation to the module plane and the Kalman operation on it, composed from the steps of
    // PlaneSteps.h. m_solve_plane marks steps within one layer (second and later hits of an in-layer
    // path): they take the intra-layer propagation flags and solve for the path to the plane.
    // Otherwise the search has already put the parameters on the plane (propPar, compute_pars())
    // and supplies the transverse path length sPerp, so only the covariance is transported.
    using namespace plane;

    // Computed at full width, see pad_free_lanes(); only the first N_filled lanes are read out.
    pad_free_lanes(N_filled, tsPar, tsErr, tsChg, sPerp, propPar, msErr, msPar, plNrm, plDir, plPnt);

    const PropagationFlags& pf =
        m_solve_plane ? prop_config->finding_intra_layer_pflags : prop_config->finding_inter_layer_pflags;
    const TrackRef in{tsPar, tsChg, NN};
    const PlaneRef pl{plPnt, plNrm};

    FieldAt f;
    field_at_start(in, pf, f);
    StartTrig t;
    start_trig(in, t);
    MPlexQF s;
    if (m_solve_plane) {
      PathSolve p;
      path_solve(in, t, pl, f, p);
      s = p.s;
      drift(in, t, f, s, propPar);
    } else {
      path_from_perp(in, t, sPerp, s);
    }
    MPlexLL J{0.0f};
    jacobian(in, t, propPar, f, s, J);
    transport_cov(J, tsErr, propErr);
    if (pf.apply_material) {
      MaterialAt m;
      material_grid(*pf.tracker_info, propPar, nullptr, NN, m);
      eloss_sign_from_path(s, nullptr, NN, m);
      apply_material(m, plNrm, propErr, propPar, NN);
    }
    outFailFlag.setVal(0);
    finish(in, tsErr, outFailFlag, propPar, propErr);

#ifdef MKFIT_TRACE_KALMAN_DEBUG
    // Charge as it goes into the update. Propagation does not change it, but the
    // update can -- through a curvature flip, formalized by
    // kalmanCheckChargeFlip() below -- and a flip is important information, so
    // keep the pre-update value for propagated_state and let updated_state carry
    // the post-update one. Comparing the two then shows the flip.
    const MPlexQI chg_pre_update = tsChg;
#endif

    const float acc_scale2 = m_chi2_trk_fac * m_chi2_trk_fac;
    const bool acc_separate = acc_scale2 != 1.0f;

    const TrackRef pred{propPar, tsChg, NN};
    MPlexQF bFld;
    local_field(pred, false, bFld);
    LocalPred L;
    to_local(pred, propErr, pl, plDir, bFld, L);
    LocalMeas M;
    measurement(L, msPar, msErr, plPnt, M);
    Residual R;
    residual(L, M, R);
    chi2(R, tsChi2);
    // det V is read only by the likelihood score.
    tsDetV = R.det;
    if (acc_separate)
      chi2_scaled(L, M, R, acc_scale2, tsChi2Acc);
    else
      tsChi2Acc = tsChi2;
    LocalUpd U;
    update(L, R, U);
    to_global(U, L, pl, tsChg, bFld, tsPar, tsErr);
    kalmanCheckChargeFlip(tsPar, tsChg, NN);

    for (int i = 0; i < N_filled; ++i) {
#ifdef MKFIT_TRACE
      // Create TrKalmanUpdate for EVERY hit that went through Kalman
      int hm_id = tr_hitmatch_ids[i];
      TrHitMatch& hm = mp_event->tr_hitmatch(hm_id);

      TrKalmanUpdate& ku = mp_event->trace_kalmanupdate({-1, hm_id, hm.state_id});
      ku.chi2 = tsChi2[i];
      ku.chi2_acc = tsChi2Acc[i];
      // accepted and state_id_out are set when a path is materialised.

#ifdef MKFIT_TRACE_KALMAN_DEBUG
      // Full pre- and post-update states, for every hit including the rejected
      // ones (for an accepted hit the post-update state is also reachable as
      // trCandStates_[state_id_out].state).
      // Charges straddle the update on purpose: pre-update on the propagated
      // state, post-update on the updated one.
      propPar.copyOut(i, ku.propagated_state.parArray_nc());
      propErr.copyOut(i, ku.propagated_state.errArray_nc());
      ku.propagated_state.charge = chg_pre_update[i];

      tsPar.copyOut(i, ku.updated_state.parArray_nc());
      tsErr.copyOut(i, ku.updated_state.errArray_nc());
      ku.updated_state.charge = tsChg[i];
#endif

      // Link forward from HitMatch
      hm.kalman_id = ku.id;
#endif

      if (mp_out) {
        ItemOut o;
        o.cand = candp[i];
        o.parent_idx = parent_idx[i];
        o.hit_pos = hit_pos[i];
        o.hot = hot[i];
        o.chi2 = tsChi2[i];
        o.det_v = tsDetV[i];
        o.chi2_acc = tsChi2Acc[i];
        tsPar.copyOut(i, o.state.parArray_nc());
        tsErr.copyOut(i, o.state.errArray_nc());
        o.state.charge = tsChg[i];
        o.state.valid = true;
#ifdef MKFIT_TRACE
        o.tr_hitmatch_id = tr_hitmatch_ids[i];
#endif
        mp_out->push_back(o);
      }
    }
  }

}  // namespace mkfit
