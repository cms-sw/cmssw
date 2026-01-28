#ifndef L1TriggerScouting_Utilities_scales_h
#define L1TriggerScouting_Utilities_scales_h

#include <cstdint>
#include <cmath>

namespace l1ScoutingRun3 {

  // Scaled used to convert scouting hw values to physical quantities

  namespace ugmt {
    struct scales {
      static constexpr float pt_scale = 0.5;
      static constexpr float ptunconstrained_scale = 1.0;
      static constexpr float phi_scale = 2. * M_PI / 576.;
      static constexpr float eta_scale = 0.0870 / 8;
      static constexpr float phi_range = M_PI;
    };
  }  // namespace ugmt

  namespace demux {
    struct scales {
      static constexpr float phi_scale = 0.0435;
      static constexpr float eta_scale = 0.0435;
      static constexpr float et_scale = 0.5;
    };
  }  // namespace demux

  namespace calol2 {
    struct scales {
      static constexpr float phi_scale = 0.0870;
      static constexpr float eta_scale = 0.0870;
      static constexpr float et_scale = 0.5;
    };
  }  // namespace calol2

}  // namespace l1ScoutingRun3
#endif  // L1TriggerScouting_Utilities_scales_h
