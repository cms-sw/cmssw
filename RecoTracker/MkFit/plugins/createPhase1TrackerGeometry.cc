//-------------------
// Phase1 tracker geometry
//-------------------

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <functional>

using namespace mkfit;

namespace {
#include "createPhase1TrackerGeometryAutoGen.acc"
}  // namespace

namespace mkfit {
  void createPhase1TrackerGeometry(TrackerInfo &ti, bool verbose) {
    ti.create_layers(18, 27, 27);
    createPhase1TrackerGeometryAutoGen(ti);

    // TODO: replace with MessageLogger
    if (verbose) {
      printf("==========================================================================================\n");
      printf("Phase1 tracker -- Create_TrackerInfo finished\n");
      printf("==========================================================================================\n");
      for (int ii = 0; ii < ti.n_layers(); ++ii)
        ti.layer(ii).print_layer();
      printf("==========================================================================================\n");
    }
  }
}  // namespace mkfit
