#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPFASTLINEARCOMMONMODENOISESUBTRACTION_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPFASTLINEARCOMMONMODENOISESUBTRACTION_H
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

class FastLinearCMNSubtractor : public SiStripCommonModeNoiseSubtractor {

  friend class SiStripRawProcessingFactory;

 public:  

  void subtract(const uint32_t&,const uint16_t&, std::vector<int16_t>&);
  void subtract(const uint32_t&,const uint16_t&, std::vector<float>&);

 private:

  template<typename T> void subtract_(const uint32_t&, const uint16_t&, std::vector<T>&);
  FastLinearCMNSubtractor(){};

};
#endif
