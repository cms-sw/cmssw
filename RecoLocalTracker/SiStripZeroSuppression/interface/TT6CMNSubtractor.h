#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPTT6COMMONMODENOISESUBTRACTION_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPTT6COMMONMODENOISESUBTRACTION_H
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

#include "FWCore/Framework/interface/ESHandle.h"
class SiStripNoises;
class SiStripQuality;

class TT6CMNSubtractor : public SiStripCommonModeNoiseSubtractor {
  
  friend class SiStripRawProcessingFactory;
  
 public:
  
  void init(const edm::EventSetup& es) override;
  void subtract(const uint32_t&, const uint16_t&, std::vector<int16_t>&) override;
  void subtract(const uint32_t&, const uint16_t&, std::vector<float>&) override;
  
 private:

  template<typename T >void subtract_(const uint32_t&,  const uint16_t&, std::vector<T>&);

  TT6CMNSubtractor(double in) : 
    cut_to_avoid_signal_(in),
    noise_cache_id(0),
    quality_cache_id(0) {};
  double cut_to_avoid_signal_;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  uint32_t noise_cache_id, quality_cache_id;
  
};
#endif
