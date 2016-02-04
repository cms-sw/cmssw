
#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPITERATEDMEDIANCOMMONMODENOISESUBTRACTION_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPITERATEDMEDIANCOMMONMODENOISESUBTRACTION_H
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

#include "FWCore/Framework/interface/ESHandle.h"
 
class SiStripNoises;
class SiStripQuality;

class IteratedMedianCMNSubtractor : public SiStripCommonModeNoiseSubtractor {
  
  friend class SiStripRawProcessingFactory;
  
 public:
  
  void init(const edm::EventSetup& es);
  void subtract(const uint32_t&,const uint16_t&, std::vector<int16_t>&);
  void subtract(const uint32_t&,const uint16_t&, std::vector<float>&);
  
 private:

  template<typename T >void subtract_(const uint32_t&, const uint16_t&, std::vector<T>&);
inline float pairMedian( std::vector<std::pair<float,float> >& sample);
 
  IteratedMedianCMNSubtractor(double sigma, int iterations) : 
    cut_to_avoid_signal_(sigma),
    iterations_(iterations),
    noise_cache_id(0),
    quality_cache_id(0) {};
  double cut_to_avoid_signal_;
  int iterations_;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  uint32_t noise_cache_id, quality_cache_id;
    

};
#endif

