#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPCOMMONMODENOISESUBTRACTOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPCOMMONMODENOISESUBTRACTOR_H

#include "FWCore/Framework/interface/EventSetup.h"
#include <vector>
#include <algorithm>
#include <cstdint>

class SiStripCommonModeNoiseSubtractor {

  friend class SiStripRawProcessingFactory;

 public:

  virtual ~SiStripCommonModeNoiseSubtractor() {};
  virtual void init(const edm::EventSetup& es) {};
  virtual void subtract(uint32_t detId, uint16_t firstStrip, std::vector<int16_t>& digis) = 0;
  virtual void subtract(uint32_t detId, uint16_t firstStrip, std::vector<float>& digis) = 0;

  const std::vector< std::pair<short,float> >& getAPVsCM() const { return _vmedians; }

 protected:

  SiStripCommonModeNoiseSubtractor(){};
  template<typename T> float median(std::vector<T>&);

  std::vector< std::pair<short,float> > _vmedians;
};

template<typename T>
inline
float SiStripCommonModeNoiseSubtractor::
median( std::vector<T>& sample) {
  typename std::vector<T>::iterator mid = sample.begin() + sample.size()/2;
  std::nth_element(sample.begin(), mid, sample.end());
  if( sample.size() & 1 ) //odd size
    return *mid;
  return ( *std::max_element(sample.begin(), mid) + *mid ) / 2.;
}
#endif
