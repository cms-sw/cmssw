#ifndef RecoTracker_MkFitCore_interface_PropagationConfig_h
#define RecoTracker_MkFitCore_interface_PropagationConfig_h

namespace mkfit {

  class TrackerInfo;

  enum PropagationFlagsEnum {
    PF_none = 0,
    PF_use_param_b_field = 0x1,
    PF_apply_material = 0x2,
    PF_copy_input_state_on_fail = 0x4
  };

  struct PropagationFlags {
    const TrackerInfo *tracker_info = nullptr;  // back-pointer for easy passing into low-level funcs
    bool use_param_b_field : 1;
    bool apply_material : 1;
    bool copy_input_state_on_fail : 1;
    // Could add: bool use_trig_approx       -- now Config::useTrigApprox = true
    // Could add: int  n_prop_to_r_iters : 8 -- now Config::Niter = 5

    PropagationFlags() : use_param_b_field(false), apply_material(false), copy_input_state_on_fail(false) {}

    PropagationFlags(int pfe)
        : use_param_b_field(pfe & PF_use_param_b_field),
          apply_material(pfe & PF_apply_material),
          copy_input_state_on_fail(pfe & PF_copy_input_state_on_fail) {}
  };

  class PropagationConfig {
  public:
    bool backward_fit_to_pca = false;
    bool finding_requires_propagation_to_hit_pos = false;
    PropagationFlags finding_inter_layer_pflags;
    PropagationFlags finding_intra_layer_pflags;
    PropagationFlags backward_fit_pflags;
    PropagationFlags forward_fit_pflags;
    PropagationFlags seed_fit_pflags;
    PropagationFlags pca_prop_pflags;

    void apply_tracker_info(const TrackerInfo *ti);
  };
}  // namespace mkfit

#endif
