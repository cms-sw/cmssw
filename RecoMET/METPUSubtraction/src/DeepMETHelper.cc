#include "RecoMET/METPUSubtraction/interface/DeepMETHelp.h"
#include <cmath>

namespace deepmet_helper {
  float scale_and_rm_outlier(float val, float scale) {
    float ret_val = val * scale;
    if (std::isnan(ret_val) || ret_val > 1e6 || ret_val < -1e6)
      return 0.;
    return ret_val;
  }

  float rm_outlier(float val) {
    if (std::isnan(val) || val > 1e6 || val < -1e6)
      return 0.;
    return val;
  }
}  // namespace deepmet_helper
