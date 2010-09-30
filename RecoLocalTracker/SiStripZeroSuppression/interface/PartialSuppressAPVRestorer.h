#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_PARTIALSUPPRESSAPVRESTORER_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_PARTIALSUPPRESSAPVRESTORER_H

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripAPVRestorer.h"

class PartialSuppressAPVRestorer : public SiStripAPVRestorer {

 friend class SiStripRawProcessingFactory;

 public:
   
  void init(const edm::EventSetup& es);
  int16_t inspect(const uint32_t&, std::vector<int16_t>&);
  int16_t inspect(const uint32_t&, std::vector<float>&);
  void restore(std::vector<int16_t>&);
  void restore(std::vector<float>&);

 private:

  PartialSuppressAPVRestorer( double fr, int dev) : 
    fraction_(fr),
    deviation_(dev){};

  template<typename T >int16_t inspect_(const uint32_t&,std::vector<T>&);
  template<typename T >void restore_(std::vector<T>&);
  
  edm::ESHandle<SiStripQuality> qualityHandle;
  uint32_t  quality_cache_id;

  double fraction_; // fraction of strips deviating from nominal baseline
  int deviation_;   // ADC value of deviation from nominal baseline 
};

#endif
