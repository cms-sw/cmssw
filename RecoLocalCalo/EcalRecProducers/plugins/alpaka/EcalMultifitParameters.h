#ifndef RecoLocalcalo_EcalRecProducers_plugins_alpaka_EcalMultifitParameters_h
#define RecoLocalcalo_EcalRecProducers_plugins_alpaka_EcalMultifitParameters_h

#include <array>

struct EcalMultifitParameters {
  static constexpr size_t kNTimeFitParams = 8;
  static constexpr size_t kNAmplitudeFitParams = 2;
  using TimeFitParamsArray = std::array<float, kNTimeFitParams>;
  using AmplitudeFitParamsArray = std::array<float, kNAmplitudeFitParams>;

  TimeFitParamsArray timeFitParamsEB;
  TimeFitParamsArray timeFitParamsEE;
  AmplitudeFitParamsArray amplitudeFitParamsEB;
  AmplitudeFitParamsArray amplitudeFitParamsEE;
};

#endif
