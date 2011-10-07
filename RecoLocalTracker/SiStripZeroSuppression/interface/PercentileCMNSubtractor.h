#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPERCENTILECOMMONMODENOISESUBTRACTION_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPERCENTILECOMMONMODENOISESUBTRACTION_H
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

class PercentileCMNSubtractor : public SiStripCommonModeNoiseSubtractor {
  
  friend class SiStripRawProcessingFactory;
  
 public:
  
  void subtract(const uint32_t&,const uint16_t& firstAPV, std::vector<int16_t>&);
  void subtract(const uint32_t&,const uint16_t& firstAPV, std::vector<float>&);
  
 private:
  
  template<typename T> float percentile(std::vector<T>&, double);
  template<typename T> void subtract_(const uint32_t&,const uint16_t& firstAPV, std::vector<T>&);
  PercentileCMNSubtractor(double in) : 
    percentile_(in) {};  
  double percentile_;
  
};
#endif
