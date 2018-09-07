#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPFASTLINEARCOMMONMODENOISESUBTRACTION_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPFASTLINEARCOMMONMODENOISESUBTRACTION_H
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

class FastLinearCMNSubtractor : public SiStripCommonModeNoiseSubtractor {

  friend class SiStripRawProcessingFactory;

 public:

  void subtract(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& digis) override;
  void subtract(uint32_t detId, uint16_t firstAPV, std::vector<float>& digis) override;

 private:

  template<typename T> void subtract_(uint32_t detId, uint16_t firstAPV, std::vector<T>& digis);
  FastLinearCMNSubtractor(){};

};
#endif
