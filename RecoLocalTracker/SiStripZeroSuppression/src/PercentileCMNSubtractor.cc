#include "RecoLocalTracker/SiStripZeroSuppression/interface/PercentileCMNSubtractor.h"

void PercentileCMNSubtractor::subtract(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& digis) {subtract_(detId, firstAPV, digis);}
void PercentileCMNSubtractor::subtract(uint32_t detId, uint16_t firstAPV, std::vector<float>& digis) {subtract_(detId,firstAPV, digis);}

template<typename T>
inline
void PercentileCMNSubtractor::
subtract_(uint32_t detId, uint16_t firstAPV, std::vector<T>& digis){

  std::vector<T> tmp;  tmp.reserve(128);
  typename std::vector<T>::iterator
    strip( digis.begin() ),
    end(   digis.end()   ),
    endAPV;

  _vmedians.clear();

  while( strip < end ) {
    endAPV = strip+128; tmp.clear();
    tmp.insert(tmp.end(),strip,endAPV);
    const float offset = percentile(tmp,percentile_);

    _vmedians.push_back(std::pair<short,float>((strip-digis.begin())/128+firstAPV,offset));

    while (strip < endAPV) {
      *strip = static_cast<T>(*strip-offset);
      strip++;
    }

  }
}


template<typename T>
inline
float PercentileCMNSubtractor::
percentile( std::vector<T>& sample, double pct) {
  typename std::vector<T>::iterator mid = sample.begin() + int(sample.size()*pct/100.0);
  std::nth_element(sample.begin(), mid, sample.end());
  return *mid;
}
