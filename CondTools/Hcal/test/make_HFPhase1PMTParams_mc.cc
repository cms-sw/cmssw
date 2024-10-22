#include "CondTools/Hcal/interface/make_HFPhase1PMTParams.h"

std::unique_ptr<HFPhase1PMTParams> make_HFPhase1PMTParams_mc() {
  // Not useful right now. Returns dummy parameters.
  return make_HFPhase1PMTParams_dummy();
}
