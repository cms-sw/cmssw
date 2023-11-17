#ifndef DF_S_SCALES_H
#define DF_S_SCALES_H

#include <cstdint>
#include <cmath>

namespace scoutingRun3 {

namespace ugmt {
    // struct gmt_scales{
    struct scales {
        static constexpr float pt_scale              = 0.5;
        static constexpr float ptunconstrained_scale = 1.0;
        static constexpr float phi_scale             = 2.*M_PI/576.;
        static constexpr float eta_scale             = 0.0870/8;        // 9th MS bit is sign
        static constexpr float phi_range             = M_PI;
    };
}



namespace demux {
    // struct gmt_scales{
    struct scales {
        static constexpr float phi_scale = 2.*M_PI/144.;
        static constexpr float eta_scale = 0.0435;
        static constexpr float et_scale  = 0.5;
    };
}

}
#endif
