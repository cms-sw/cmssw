#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitPhase2WeightsStruct_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitPhase2WeightsStruct_h

#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // define a struct for the data
  struct EcalUncalibRecHitPhase2Weights {
    std::array<double, ecalPh2::sampleSize> weights;
    std::array<double, ecalPh2::sampleSize> timeWeights;
  };
}  //namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsStruct_h
