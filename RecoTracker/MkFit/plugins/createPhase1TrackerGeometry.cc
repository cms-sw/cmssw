//-------------------
// Phase1 tracker geometry
//-------------------

#include "Config.h"
#include "Debug.h"
#include "TrackerInfo.h"

#include <functional>

using namespace mkfit;

namespace {
#include "createPhase1TrackerGeometryAutoGen.acc"
}  // namespace

namespace mkfit {
  void createPhase1TrackerGeometry(TrackerInfo &ti, bool verbose) {
    // TODO: these writes to global variables need to be removed
    Config::nTotalLayers = 18 + 2 * 27;

    Config::useCMSGeom = true;

    Config::finding_requires_propagation_to_hit_pos = true;
    Config::finding_inter_layer_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    Config::finding_intra_layer_pflags = PropagationFlags(PF_none);
    Config::backward_fit_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    Config::forward_fit_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    Config::seed_fit_pflags = PropagationFlags(PF_none);
    Config::pca_prop_pflags = PropagationFlags(PF_none);

    ti.set_eta_regions(0.9, 1.7, 2.45, false);
    ti.create_layers(18, 27, 27);
    createPhase1TrackerGeometryAutoGen(ti);

    // TODO: replace with MessageLogger
    if (verbose) {
      printf("==========================================================================================\n");
      printf("Phase1 tracker -- Create_TrackerInfo finished\n");
      printf("==========================================================================================\n");
      for (auto &i : ti.m_layers)
        i.print_layer();
      printf("==========================================================================================\n");
    }
  }
}  // namespace mkfit
