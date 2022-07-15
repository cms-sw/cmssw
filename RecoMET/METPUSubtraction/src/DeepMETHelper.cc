#include "RecoMET/METPUSubtraction/interface/DeepMETHelp.h"

namespace deepmet_helper {
  float scale_and_rm_outlier(float val, float scale) {
    float ret_val = val * scale;
    if (ret_val > 1e6 || ret_val < -1e6)
      return 0.;
    return ret_val;
  }

}  // namespace deepmet_helper
