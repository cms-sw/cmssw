#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPERCENTILECOMMONMODENOISESUBTRACTION_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPERCENTILECOMMONMODENOISESUBTRACTION_H
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

class PercentileCMNSubtractor : public SiStripCommonModeNoiseSubtractor {

  friend class SiStripRawProcessingFactory;

 public:

  void subtract(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& digis) override;
  void subtract(uint32_t detId, uint16_t firstAPV, std::vector<float>&   digis) override;

 private:

  template<typename T> float percentile(std::vector<T>&, double);
  template<typename T> void subtract_(uint32_t detId, uint16_t firstAPV, std::vector<T>& digis);
  PercentileCMNSubtractor(double in) :
    percentile_(in) {};
  double percentile_;

};
#endif
