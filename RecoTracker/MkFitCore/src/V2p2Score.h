#ifndef RecoTracker_MkFitCore_src_V2p2Score_h
#define RecoTracker_MkFitCore_src_V2p2Score_h

#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/src/V2p2Config.h"

#include <cmath>

namespace mkfit {

  //============================================================================
  // The MkFinderV2p2 layer-step score. Separate from track_score_func, which
  // V1/V2 and the final output keep using. It is additive over layer steps, has
  // no tail-hole concept, and takes a feature struct. See
  // doc/MkFinderV2p2-DesignNotes.md, "Layer-step score".
  //============================================================================

  // Why a candidate took no hit in this layer. Mirrors the Hit::kHit*Idx values
  // that get recorded, but as a dense enum so it can index a parameter table.
  enum V2p2HoleKind_e {
    V2P2_NoHole = 0,  // hits were taken
    V2P2_Miss,        // sensitive region crossed, nothing found -- a real hole
    V2P2_Edge,        // crossing within dq of the layer boundary
    V2P2_Gap,         // crossing in an endcap r hole
    V2P2_Stop,        // out of hole budget
    V2P2_HoleKind_Count
  };

  // One layer step: everything that happened to one candidate between its state
  // at layer check-in and its state at the end of the layer. Kept a POD so it can
  // go into the trace unchanged. A term that needs a new input adds a member: a
  // fact of the step or of the candidate is filled in
  // MkFinderV2p2::fill_step_geometry(), a per-hit one accumulated in
  // MkFinderV2p2::harvest_nodes().
  struct LayerStepFeatures {
    // What was taken.
    float chi2_sum = 0.0f;  // over the hits of this step
    // Sum over the hits taken of ln(det V), V the 2x2 residual covariance chi2 is
    // formed with. Carries the hit precision that chi2 divides out.
    float log_det_v_sum = 0.0f;
    // ln of the local hit density in the search window, hits per cm^2.
    float log_rho = 0.0f;
    short n_hits = 0;     // hits taken in this layer: 0, 1, 2, ...
    short n_overlap = 0;  // = max(n_hits - 1, 0)
    short hole_kind = V2P2_NoHole;

    // Where.
    short layer_to = -1;
    bool is_outward = true;
    bool is_pixel = false;
    bool is_barrel = true;
  };

  // The coefficients are V2p2ScoreParams, interface/V2p2Params.h; the score mode
  // and the term switches are in V2p2Config.h, Config::V2p2::Score.

  // Layer group of V2p2ScoreParams::hit_eff_grp. The barrel strip split at layer 10 is the
  // phase-2 boundary between the OT PS and 2S barrel (layers 4-9 and 10-15); it means nothing for phase 1,
  // where the outward defaults of V2p2Params, a phase-2 tuning, then apply to the
  // TIB / TOB layers below and above layer 10.
  inline int v2p2_hit_eff_group(const LayerStepFeatures& f) {
    if (f.is_pixel)
      return f.is_barrel ? 0 : 1;
    if (f.is_barrel)
      return f.layer_to < 10 ? 2 : 3;
    return 4;
  }

  // Log-likelihood ratio against "this layer produced no hit":
  //   take hit j :  ln(eps/(1-eps)) - ln(2pi) - chi2_j/2 - ln(det V_j)/2 - ln rho
  //   take none  :  0
  // eps = hit_eff, V the 2x2 residual covariance, rho the local hit density.
  inline float v2p2_layer_step_loglh(const LayerStepFeatures& f, const V2p2ScoreParams& p) {
    if (f.n_hits == 0)
      return 0.0f;  // the reference hypothesis, whatever kind of hole it was
    const float eps_grp = p.hit_eff_grp[v2p2_hit_eff_group(f)];
    const float eps = eps_grp >= 0.0f ? eps_grp : p.hit_eff;
    const float c_eps = std::log(eps / (1.0f - eps)) - 1.8378771f;  // ln(2pi)
#if defined(MKFIT_STANDALONE)
    namespace ss = Config::V2p2::ScoreStats;
    if (ss::accum) {
      ss::n_hits += f.n_hits;
      ss::sum_log_rho += (double)f.n_hits * f.log_rho;
      ss::sum_log_detv += f.log_det_v_sum;
    }
#endif
    namespace sc = Config::V2p2::Score;
    // log_rho is per layer STEP, so it multiplies n_hits; log_det_v_sum is
    // already a sum over the hits taken, so its replacement must be scaled.
    const float lrho = sc::use_rho ? f.log_rho : sc::rho_const;
    const float ldetv = sc::use_detv ? f.log_det_v_sum : f.n_hits * sc::detv_const;
    return f.n_hits * (c_eps - lrho) - 0.5f * (f.chi2_sum + ldetv);
  }

  inline float v2p2_layer_step_score(const LayerStepFeatures& f, const V2p2Params& vp) {
    namespace sc = Config::V2p2::Score;
    const V2p2ScoreParams& p = vp.score(f.is_outward);

    if (sc::mode == 1)
      return v2p2_layer_step_loglh(f, p);

    float s = p.hit_bonus * f.n_hits + p.overlap_bonus * f.n_overlap - p.chi2_weight * f.chi2_sum;

    switch (f.hole_kind) {
      case V2P2_Miss:
        s -= p.miss_penalty;
        break;
      case V2P2_Edge:
        s -= p.edge_penalty;
        break;
      case V2P2_Gap:
        s -= p.gap_penalty;
        break;
      case V2P2_Stop:
        s -= p.stop_penalty;
        break;
      default:
        break;
    }
    return s;
  }

  // Hit::kHit*Idx -> V2p2HoleKind_e, so the fake-hit decision and the score stay
  // one decision made in one place.
  inline V2p2HoleKind_e v2p2_hole_kind(int fake_hit_idx) {
    switch (fake_hit_idx) {
      case Hit::kHitMissIdx:
        return V2P2_Miss;
      case Hit::kHitEdgeIdx:
        return V2P2_Edge;
      case Hit::kHitInGapIdx:
        return V2P2_Gap;
      case Hit::kHitStopIdx:
        return V2P2_Stop;
      default:
        return V2P2_Miss;
    }
  }

}  // namespace mkfit

#endif
