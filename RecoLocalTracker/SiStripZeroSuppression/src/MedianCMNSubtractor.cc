#include "RecoLocalTracker/SiStripZeroSuppression/interface/MedianCMNSubtractor.h"

void MedianCMNSubtractor::subtract(const uint32_t& detId,const uint16_t& firstAPV, std::vector<int16_t>& digis) {subtract_(detId,firstAPV,digis);}
void MedianCMNSubtractor::subtract(const uint32_t& detId,const uint16_t& firstAPV, std::vector<float>& digis) {subtract_(detId,firstAPV, digis);}

template<typename T> 
inline
void MedianCMNSubtractor::
subtract_(const uint32_t& detId,const uint16_t& firstAPV, std::vector<T>& digis){
  
  std::vector<T> tmp;  tmp.reserve(128);  
  typename std::vector<T>::iterator  
    strip( digis.begin() ), 
    end(   digis.end()   ),
    endAPV;
  
  _vmedians.clear();
  
  while( strip < end ) {
    endAPV = strip+128; tmp.clear();
    tmp.insert(tmp.end(),strip,endAPV);
    const float offset = median(tmp);

    _vmedians.push_back(std::pair<short,float>((strip-digis.begin())/128+firstAPV,offset));
    
    while (strip < endAPV) {
      *strip = static_cast<T>(*strip-offset);
      strip++;
    }

  }
}
