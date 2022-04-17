#include "RecoTracker/MkFitCore/interface/ConfigWrapper.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

namespace mkfit {
  namespace ConfigWrapper {
    void initializeForCMSSW() {
      PropagationConfig pconf;
      pconf.backward_fit_to_pca = false;
      pconf.finding_requires_propagation_to_hit_pos = true;
      pconf.finding_inter_layer_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
      pconf.finding_intra_layer_pflags = PropagationFlags(PF_none);
      pconf.backward_fit_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
      pconf.forward_fit_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
      pconf.seed_fit_pflags = PropagationFlags(PF_none);
      pconf.pca_prop_pflags = PropagationFlags(PF_none);
      pconf.set_as_default();
    }
  }  // namespace ConfigWrapper
}  // namespace mkfit
