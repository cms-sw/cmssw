#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPCOMMONMODENOISESUBTRACTOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPCOMMONMODENOISESUBTRACTOR_H

#include "FWCore/Framework/interface/EventSetup.h"
#include <vector>
#include <algorithm>
#include <stdint.h>

class SiStripCommonModeNoiseSubtractor {

  friend class SiStripRawProcessingFactory;

 public:
  
  virtual ~SiStripCommonModeNoiseSubtractor() {};
  virtual void init(const edm::EventSetup& es) {};
  virtual void subtract(const uint32_t&, const uint16_t&, std::vector<int16_t>&) = 0;
  virtual void subtract(const uint32_t&, const uint16_t&, std::vector<float>&) = 0;
  
  const std::vector< std::pair<short,float> >& getAPVsCM(){return _vmedians;}
  
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
