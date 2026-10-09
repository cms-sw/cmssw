#ifndef RecoTracker_MkFitCore_interface_PropagationConfig_h
#define RecoTracker_MkFitCore_interface_PropagationConfig_h

namespace mkfit {

  class TrackerInfo;

  enum PropagationFlagsEnum {
    PF_none = 0,
    PF_use_param_b_field = 0x1,
    PF_apply_material = 0x2,
    PF_copy_input_state_on_fail = 0x4,
    PF_use_fit_b_field = 0x8
  };

  class PropagationFlags {
  public:
    const TrackerInfo *tracker_info = nullptr;  // back-pointer for easy passing into low-level funcs
    bool use_param_b_field : 1;
    bool apply_material : 1;
    bool copy_input_state_on_fail : 1;
    bool use_fit_b_field : 1;  // the parametrised field of the final fit (Config::bFieldFromZRFit)
    // Could add: bool use_trig_approx       -- now Config::useTrigApprox = true
    // Could add: int  n_prop_to_r_iters : 8 -- now Config::Niter = 5

    PropagationFlags()
        : use_param_b_field(false), apply_material(false), copy_input_state_on_fail(false), use_fit_b_field(false) {}

    PropagationFlags(int pfe)
        : use_param_b_field(pfe & PF_use_param_b_field),
          apply_material(pfe & PF_apply_material),
          copy_input_state_on_fail(pfe & PF_copy_input_state_on_fail),
          use_fit_b_field(pfe & PF_use_fit_b_field) {}
  };

  // Choices of the final fit, MkBuilder::fit_tracks(), read by its own sequences (FinalFit.cc).
  struct FinalFitFlags {
    // Sample B at the chord midpoint of each propagation to a plane instead of at its start, so that the
    // outward and inward propagations are inverses of each other.  With the parametrised field only.
    bool b_field_at_mid = true;
    // Correct each propagation to a plane for the radial field component Br = -(r/2) dBz/dz, which the
    // constant-Bz helix neglects, antisymmetrically: half of the change in r*p_phi at each end of the step.
    // With the parametrised field only.
    bool radial_field_corr = true;
    // Sign of the energy loss from the fit pass, not from the sign of each step's path length: the forward
    // pass loses energy on every step, the backward pass gains it.  The particle crosses every module whatever
    // order the fit visits them in; the path-length sign is wrong on every step the fit takes backwards, e.g.
    // between the two sensors of a PS module visited in reverse order.
    bool eloss_sign_from_pass = true;
    // Multiple-scattering noise of the backward pass at the |p| of its start state (the forward result), fixed
    // per track, instead of at the running estimate, which correlates the assumed noise with the estimate's own
    // error.  Only theta0 changes; the geometric factors and the energy loss use the current state.
    bool bkw_ms_fixed_momentum = true;
    // Number of sub-steps of each propagation of the backward pass (1 = one step).  The parameters are carried
    // through the sub-steps, each with the field model of a full step; the covariance is transported once, with
    // the Jacobian of the whole step, and material is applied at the destination.
    int bkw_sub_steps = 2;
    // Material of each crossed module from its own MediumProperties (ModuleInfo::radl, bbxi) instead of the
    // (|z|, r) grid, which averages over the modules overlapping a cell and has no phi dimension.
    bool material_per_module = true;
  };

  class PropagationConfig {
  public:
    bool backward_fit_to_pca = false;
    bool finding_requires_propagation_to_hit_pos = false;
    PropagationFlags finding_inter_layer_pflags;
    PropagationFlags finding_intra_layer_pflags;
    PropagationFlags backward_fit_pflags;
    // The final fit, MkBuilder::fit_tracks(), both passes.
    PropagationFlags final_fit_pflags;
    FinalFitFlags final_fit_ffflags;
    PropagationFlags seed_fit_pflags;
    PropagationFlags pca_prop_pflags;

    void apply_tracker_info(const TrackerInfo *ti);
  };
}  // namespace mkfit

#endif
