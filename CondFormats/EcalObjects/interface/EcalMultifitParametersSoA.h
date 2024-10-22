#ifndef CondFormats_EcalObjects_EcalMultifitParametersSoA_h
#define CondFormats_EcalObjects_EcalMultifitParametersSoA_h

#include <array>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

constexpr size_t kNTimeFitParams = 8;
constexpr size_t kNAmplitudeFitParams = 2;
using TimeFitParamsArray = std::array<float, kNTimeFitParams>;
using AmplitudeFitParamsArray = std::array<float, kNAmplitudeFitParams>;

GENERATE_SOA_LAYOUT(EcalMultifitParametersSoALayout,
                    SOA_SCALAR(TimeFitParamsArray, timeFitParamsEB),
                    SOA_SCALAR(TimeFitParamsArray, timeFitParamsEE),
                    SOA_SCALAR(AmplitudeFitParamsArray, amplitudeFitParamsEB),
                    SOA_SCALAR(AmplitudeFitParamsArray, amplitudeFitParamsEE))

using EcalMultifitParametersSoA = EcalMultifitParametersSoALayout<>;

#endif
