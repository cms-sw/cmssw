#ifndef RecoTracker_MkFitCore_interface_V2p2Params_h
#define RecoTracker_MkFitCore_interface_V2p2Params_h

namespace mkfit {

  // Coefficients of the layer-step score (src/V2p2Score.h), one set per search
  // direction.
  struct V2p2ScoreParams {
    // hit_bonus .. stop_penalty: linear mode only.
    float hit_bonus = 3.0f;      // per hit taken; a hole wins above chi2 11, see the design notes
    float overlap_bonus = 0.0f;  // per hit beyond the first in a layer
    float chi2_weight = 1.0f;    // per unit of chi2
    float miss_penalty = 8.0f;   // a real hole
    float edge_penalty = 0.0f;   // crossing at the layer boundary
    float gap_penalty = 0.0f;    // crossing in an inactive region
    float stop_penalty = 8.0f;   // out of hole budget
    float hit_eff = 0.90f;       // per-layer hit efficiency, likelihood mode only
    // Per layer group, likelihood mode only; a negative value takes hit_eff. Groups, see
    // v2p2_hit_eff_group(): 0 pixel barrel, 1 pixel endcap, 2 strip barrel below layer 10
    // (phase-2 OT PS barrel), 3 strip barrel from layer 10 (OT 2S barrel), 4 strip endcap.
    static constexpr int n_hit_eff_grp = 5;
    float hit_eff_grp[n_hit_eff_grp] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  };

  // The MkFinderV2p2 tunings that can be set per iteration. In CMSSW they are
  // IterationConfig::m_v2p2, set by MkFitIterationConfigESProducer from its python
  // PSet "v2p2" and not saved to or loaded from JSON; of the score sets the PSet
  // carries hit_eff and hit_eff_grp, the fields the log-likelihood score reads. In
  // the standalone the mkFit options and the shell setters change them in
  // Config::ItrInfo. The values below are the defaults of both.
  struct V2p2Params {
    // Outward: a phase-2 tuning (CMSSW MTV, initialStep with the offline seeds, cleaner and
    // refit, 2026-10): eps 0.10, OT PS barrel 0.14, OT 2S barrel 0.20. It will move with the seeds and the
    // final fit.
    V2p2ScoreParams score_outward{.hit_eff = 0.10f, .hit_eff_grp = {-1.0f, -1.0f, 0.14f, 0.20f, -1.0f}};
    V2p2ScoreParams score_inward{};

    // Kalman hit acceptance: a hit is accepted if
    //   chi2_acc = r^T (chi2_trk_fac^2 C_trk + C_hit)^-1 r < hit_chi2_cut,
    // with r the residual, all in the module's local 2-D frame. chi2_trk_fac carries the
    // deficit of the track covariance, so hit_chi2_cut is a chi2 quantile: 9.21 is the
    // 99 % point with 2 dof. chi2_trk_fac = 1 is the plain chi2. The Kalman update and the
    // score use the plain chi2. v2p2 does not use IterationParams::chi2Cut_min.
    float chi2_trk_fac = 3.0f;
    float hit_chi2_cut = 9.21f;

    int max_presel_hits = 6;     // reduction cap per sub-layer
    int max_hits_per_layer = 4;  // most hits one candidate may take in one plan entry (its in-layer path)

    static constexpr int max_presel_hits_limit = 12;    // inline storage
    static constexpr int max_hits_per_layer_limit = 8;  // array size

    const V2p2ScoreParams& score(bool is_outward) const { return is_outward ? score_outward : score_inward; }
  };

}  // namespace mkfit

#endif
